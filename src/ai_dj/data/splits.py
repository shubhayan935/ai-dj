"""End-to-end dataset builder: join, clean, rank, DJ-disjoint split, and export SFT/DPO JSONL."""

from __future__ import annotations

import bisect
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from ai_dj.data.parsing import (
    classify_set_phase,
    extract_genres,
    normalize_name,
    parse_mix_title,
    parse_track_title,
)
from ai_dj.data.quality import (
    QualityConfig,
    assign_quality_tiers,
    classify_transition_style,
    compute_tempo_stretch_pct,
    score_transition,
)
from ai_dj.data.schemas import (
    CandidateTrackSummary,
    DPOPair,
    MixSet,
    SetBrief,
    SFTExample,
    SplitName,
    TrackCard,
    TransitionPlan,
    TransitionRecord,
)


def assign_dj_split(
    dj_name_norm: str,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
) -> SplitName:
    """Deterministically assign a DJ to train/val/test so no DJ leaks across splits."""
    digest = hashlib.sha256(dj_name_norm.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") / float(2**32)
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def _to_candidate(tc: TrackCard) -> CandidateTrackSummary:
    return CandidateTrackSummary(
        track_id=tc.track_id,
        artist=tc.artist,
        title=tc.title,
        version=tc.version,
        record_label=tc.record_label,
        bpm=tc.bpm,
        duration_sec=tc.duration_sec,
        genres=tc.genres[:5],
    )


def _degrade_plan_off_phrase(plan: TransitionPlan, rng: random.Random) -> TransitionPlan:
    """Create a realistic off-phrase / mistimed negative transition plan for DPO."""
    shift = rng.choice([2.5, 3.0, 5.0, -2.5, -3.0])
    bad_overlap = max(1.0, round(plan.overlap_beats + shift, 2))
    return plan.model_copy(
        update={
            "track_cue_in_beat_next": max(0.0, round(plan.track_cue_in_beat_next + shift, 2)),
            "overlap_beats": bad_overlap,
            "overlap_bars": round(bad_overlap / 4.0, 2),
            "transition_style": classify_transition_style(bad_overlap),
        }
    )


def load_mixes_and_tracks(
    metadata_path: Path,
    tracks_bpm_path: Path,
) -> tuple[dict[str, MixSet], dict[str, TrackCard], dict[str, list[str]]]:
    """Load djmix-dataset.json and tracks_bpm.parquet into canonical MixSet and TrackCard maps."""
    with metadata_path.open("r", encoding="utf-8") as fh:
        raw_mixes = json.load(fh)

    bpm_table = pq.read_table(tracks_bpm_path)
    bpm_map: dict[str, tuple[float | None, float | None, float | None, int | None]] = {}
    for tid, bpm, stab, dur, bc in zip(
        bpm_table.column("track_id").to_pylist(),
        bpm_table.column("bpm").to_pylist(),
        bpm_table.column("bpm_stability").to_pylist(),
        bpm_table.column("duration_sec").to_pylist(),
        bpm_table.column("beat_count").to_pylist(),
        strict=True,
    ):
        bpm_map[tid] = (bpm, stab, dur, bc)

    mixes: dict[str, MixSet] = {}
    track_play_counts: Counter[str] = Counter()
    track_mix_sets: dict[str, set[str]] = defaultdict(set)
    track_genres: dict[str, Counter[str]] = defaultdict(Counter)
    track_raw_title: dict[str, str] = {}
    mix_track_order: dict[str, list[str]] = {}

    for m in raw_mixes:
        mix_id = m["id"]
        raw_tags = m.get("tags") or []
        tags: list[str] = []
        for t in raw_tags:
            val = t.get("key", "") if isinstance(t, dict) else str(t or "")
            val = val.removeprefix("Category:").strip()
            if val:
                tags.append(val)
        parsed = parse_mix_title(m.get("title", ""), tags)
        genres = extract_genres(tags, parsed.dj_name_norm)
        num_total = int(m.get("num_total_tracks") or 0)
        num_id = int(m.get("num_identified_tracks") or 0)
        id_ratio = round(num_id / num_total, 4) if num_total > 0 else 0.0

        mixes[mix_id] = MixSet(
            mix_id=mix_id,
            title=m.get("title", ""),
            url=m.get("url"),
            audio_source=m.get("audio_source"),
            audio_url=m.get("audio_url"),
            dj_name=parsed.dj_name,
            dj_name_norm=parsed.dj_name_norm,
            venue_or_show=parsed.venue_or_show,
            mix_date=parsed.mix_date,
            year=parsed.year,
            tags=tags,
            genres=genres,
            num_total_tracks=num_total,
            num_identified_tracks=num_id,
            num_available_transitions=int(m.get("num_available_transitions") or 0),
            identification_ratio=id_ratio,
            curated_series=parsed.curated_series,
        )

        mix_split = assign_dj_split(parsed.dj_name_norm)
        ordered_ids: list[str] = []
        for entry in m.get("tracklist") or []:
            tid = entry.get("id")
            if not tid:
                continue
            ordered_ids.append(tid)
            track_play_counts[tid] += 1
            track_mix_sets[tid].add(mix_id)
            # Only aggregate track-level genres from train mixes so held-out val/test
            # mix briefs never leak into candidate track cards.
            if mix_split == "train":
                for genre in genres:
                    track_genres[tid][genre] += 1
            if tid not in track_raw_title and entry.get("title"):
                track_raw_title[tid] = entry["title"]
        mix_track_order[mix_id] = ordered_ids

    tracks: dict[str, TrackCard] = {}
    all_tids = set(track_play_counts.keys()) | set(bpm_map.keys())
    for tid in all_tids:
        raw_t = track_raw_title.get(tid, "")
        pt = parse_track_title(raw_t) if raw_t else None
        bpm, stab, dur, bc = bpm_map.get(tid, (None, None, None, None))
        top_genres = [g for g, _ in track_genres[tid].most_common(5)]
        tracks[tid] = TrackCard(
            track_id=tid,
            raw_title=raw_t,
            artist=pt.artist if pt else "Unknown",
            title=pt.title if pt else "Unknown",
            version=pt.version if pt else None,
            record_label=pt.record_label if pt else None,
            bpm=bpm,
            bpm_stability=stab,
            duration_sec=dur,
            beat_count=bc,
            genres=top_genres,
            play_count=track_play_counts[tid],
            unique_mix_count=len(track_mix_sets[tid]),
            youtube_url=f"https://www.youtube.com/watch?v={tid}",
        )

    return mixes, tracks, mix_track_order


def build_cleaned_dataset(
    raw_dir: Path,
    out_dir: Path,
    config: QualityConfig | None = None,
    candidate_pool_size: int = 20,
    history_window: int = 3,
    dj_dropout_prob: float = 0.50,
    seed: int = 42,
) -> dict:
    """Run full join -> quality filter -> tier ranking -> DJ-disjoint split -> JSONL export."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or QualityConfig()

    mixes, tracks, mix_track_order = load_mixes_and_tracks(
        raw_dir / "djmix-dataset.json",
        raw_dir / "tracks_bpm.parquet",
    )
    tran_table = pq.read_table(raw_dir / "transitions_scalar.parquet")
    rows = tran_table.to_pylist()

    # Pass 1: count transition & artist-pair repeatability across distinct mixes
    pair_mixes: dict[tuple[str, str], set[str]] = defaultdict(set)
    artist_pair_mixes: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in rows:
        tp, tn, mid = r["track_id_prev"], r["track_id_next"], r["mix_id"]
        if not tp or not tn or tp == tn:
            continue
        pair_mixes[(tp, tn)].add(mid)
        ap = normalize_name(tracks[tp].artist) if tp in tracks else "unknown"
        an = normalize_name(tracks[tn].artist) if tn in tracks else "unknown"
        if ap != "unknown" and an != "unknown":
            artist_pair_mixes[(ap, an)].add(mid)

    # Pass 2: construct & score TransitionRecords
    records: list[TransitionRecord] = []
    for r in rows:
        mid = r["mix_id"]
        tp = r["track_id_prev"]
        tn = r["track_id_next"]
        if mid not in mixes or tp not in tracks or tn not in tracks or tp == tn:
            continue

        mix = mixes[mid]
        prev_tc = tracks[tp]
        next_tc = tracks[tn]

        stretch_prev = compute_tempo_stretch_pct(
            float(r["matched_time_mix_prev"] or 0.0),
            float(r["matched_time_track_prev"] or 0.0),
        )
        stretch_next = compute_tempo_stretch_pct(
            float(r["matched_time_mix_next"] or 0.0),
            float(r["matched_time_track_next"] or 0.0),
        )
        # tran_wpts is in half-beats (2 warping points per beat) and measures the actual
        # audible transition window between cue_out_prev and cue_in_next.
        tran_wpts = float(r["tran_wpts"] or 0.0)
        overlap_beats = (
            round(tran_wpts / 2.0, 2) if tran_wpts > 0 else float(r["overlap_beats"] or 0.0)
        )
        key_prev = int(r["key_change_prev"] or 0)
        key_next = int(r["key_change_next"] or 0)

        ap = normalize_name(prev_tc.artist)
        an = normalize_name(next_tc.artist)
        pair_rep = len(pair_mixes[(tp, tn)])
        art_rep = len(artist_pair_mixes[(ap, an)]) if ap != "unknown" and an != "unknown" else 1

        quality = score_transition(
            match_rate_prev=float(r["match_rate_prev"] or 0.0),
            match_rate_next=float(r["match_rate_next"] or 0.0),
            cost_prev=float(r["cost_prev"] or 1.0),
            cost_next=float(r["cost_next"] or 1.0),
            matched_beats_prev=int(r["matched_beats_prev"] or 0),
            matched_beats_next=int(r["matched_beats_next"] or 0),
            overlap_beats=overlap_beats,
            stretch_prev_pct=stretch_prev,
            stretch_next_pct=stretch_next,
            key_change_prev=key_prev,
            key_change_next=key_next,
            bpm_prev=prev_tc.bpm,
            bpm_next=next_tc.bpm,
            stability_prev=prev_tc.bpm_stability,
            stability_next=next_tc.bpm_stability,
            mix_id_ratio=mix.identification_ratio,
            curated_series=mix.curated_series,
            pair_repeat_count=pair_rep,
            artist_pair_repeat_count=art_rep,
            config=cfg,
        )

        bpm_delta = (
            round(next_tc.bpm - prev_tc.bpm, 2)
            if prev_tc.bpm is not None and next_tc.bpm is not None
            else None
        )

        plan = TransitionPlan(
            track_cue_out_beat_prev=round(float(r["track_cue_out_beat_prev"] or 0.0), 2),
            track_cue_in_beat_next=round(float(r["track_cue_in_beat_next"] or 0.0), 2),
            overlap_beats=round(overlap_beats, 2),
            overlap_bars=round(overlap_beats / 4.0, 2),
            tempo_stretch_prev_pct=stretch_prev,
            tempo_stretch_next_pct=stretch_next,
            key_change_prev_semitones=key_prev,
            key_change_next_semitones=key_next,
            transition_style=classify_transition_style(overlap_beats),
        )

        records.append(
            TransitionRecord(
                tran_id=r["tran_id"],
                mix_id=mid,
                i_tran=int(r["i_tran"]),
                dj_name=mix.dj_name,
                dj_name_norm=mix.dj_name_norm,
                mix_year=mix.year,
                mix_tags=mix.tags,
                curated_series=mix.curated_series,
                prev_track=prev_tc,
                next_track=next_tc,
                match_rate_prev=round(float(r["match_rate_prev"] or 0.0), 4),
                match_rate_next=round(float(r["match_rate_next"] or 0.0), 4),
                cost_prev=round(float(r["cost_prev"] or 0.0), 4),
                cost_next=round(float(r["cost_next"] or 0.0), 4),
                matched_beats_prev=int(r["matched_beats_prev"] or 0),
                matched_beats_next=int(r["matched_beats_next"] or 0),
                bpm_delta=bpm_delta,
                pair_repeat_count=pair_rep,
                artist_pair_repeat_count=art_rep,
                plan=plan,
                quality=quality,
            )
        )

    # Pass 3: assign gold/silver/bronze tiers & DJ-disjoint splits
    tier_map = assign_quality_tiers([(i, rec.quality) for i, rec in enumerate(records)])
    for i, rec in enumerate(records):
        rec.quality.tier = tier_map[i]
        rec.split = assign_dj_split(rec.dj_name_norm)

    # Build BPM-sorted track indices (global + per-genre) for O(log N) hard-negative sampling
    bpm_indexed = sorted(
        [
            (tc.bpm, tid)
            for tid, tc in tracks.items()
            if tc.bpm is not None and 70.0 <= tc.bpm <= 185.0
        ]
    )
    bpm_keys = [x[0] for x in bpm_indexed]

    genre_bpm_raw: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for bpm_val, tid in bpm_indexed:
        for g in tracks[tid].genres:
            genre_bpm_raw[g.lower()].append((bpm_val, tid))
    genre_bpm_indexed: dict[str, tuple[list[float], list[str]]] = {
        g: ([x[0] for x in lst], [x[1] for x in lst]) for g, lst in genre_bpm_raw.items()
    }

    def sample_hard_distractors(
        target_bpm: float,
        target_genres: list[str],
        exclude_ids: set[str],
        k: int,
        rng: random.Random,
    ) -> list[TrackCard]:
        """Sample BPM-matched distractors with ~75% same-genre hard negatives."""
        lo_bpm = target_bpm * 0.94
        hi_bpm = target_bpm * 1.06

        chosen_ids: list[str] = []
        chosen_set = set(exclude_ids)

        # 1. Sample ~75% from BPM + genre-matched hard negatives via per-genre bisect
        n_hard_target = round(k * 0.75)
        if target_genres and n_hard_target > 0:
            genre_candidates: list[str] = []
            seen_g: set[str] = set()
            for g in target_genres:
                entry = genre_bpm_indexed.get(g.lower())
                if not entry:
                    continue
                g_keys, g_tids = entry
                glo = bisect.bisect_left(g_keys, lo_bpm)
                ghi = bisect.bisect_right(g_keys, hi_bpm)
                # Cap per-genre slice by random reservoir sampling if huge
                span = ghi - glo
                if span <= 0:
                    continue
                idxs = rng.sample(range(glo, ghi), min(span, n_hard_target * 4))
                for idx in idxs:
                    tid = g_tids[idx]
                    if tid not in chosen_set and tid not in seen_g:
                        seen_g.add(tid)
                        genre_candidates.append(tid)
            if genre_candidates:
                hard_ids = rng.sample(genre_candidates, min(n_hard_target, len(genre_candidates)))
                chosen_ids.extend(hard_ids)
                chosen_set.update(hard_ids)

        # 2. Fill remaining slots from global BPM-matched pool
        rem = k - len(chosen_ids)
        if rem > 0:
            lo = bisect.bisect_left(bpm_keys, lo_bpm)
            hi = bisect.bisect_right(bpm_keys, hi_bpm)
            span = hi - lo
            if span >= rem + len(chosen_set):
                # Rejection sample directly from [lo, hi) in O(rem)
                attempts = 0
                while len(chosen_ids) < k and attempts < rem * 20:
                    attempts += 1
                    tid = bpm_indexed[rng.randrange(lo, hi)][1]
                    if tid not in chosen_set:
                        chosen_set.add(tid)
                        chosen_ids.append(tid)
            if len(chosen_ids) < k:
                fallback = [
                    bpm_indexed[j][1]
                    for j in range(lo, hi)
                    if bpm_indexed[j][1] not in chosen_set
                ]
                if len(fallback) < k - len(chosen_ids):
                    fallback = [tid for _, tid in bpm_indexed if tid not in chosen_set]
                extra = rng.sample(fallback, min(k - len(chosen_ids), len(fallback)))
                chosen_ids.extend(extra)

        return [tracks[tid] for tid in chosen_ids]

    # Write clean transitions, SFT splits, and DPO pairs
    clean_records = [rec for rec in records if rec.quality.passed_hard_gates]
    clean_records.sort(key=lambda r: r.quality.composite_score, reverse=True)

    clean_path = out_dir / "transitions_clean.jsonl"
    with clean_path.open("w", encoding="utf-8") as fh:
        for rec in clean_records:
            fh.write(rec.model_dump_json() + "\n")

    sft_handles = {
        split: (out_dir / f"sft_{split}.jsonl").open("w", encoding="utf-8")
        for split in ("train", "val", "test")
    }
    dpo_handle = (out_dir / "dpo_train.jsonl").open("w", encoding="utf-8")

    # Precompute per-mix BPM range (p10..p90) for SetBrief conditioning
    mix_bpm_bounds: dict[str, tuple[float | None, float | None]] = {}
    for mid, tids in mix_track_order.items():
        bpms_m = sorted(
            tracks[tid].bpm
            for tid in tids
            if tid in tracks and tracks[tid].bpm is not None and 70.0 <= tracks[tid].bpm <= 185.0
        )
        if bpms_m:
            n_b = len(bpms_m)
            lo_b = round(bpms_m[int(0.10 * (n_b - 1))], 1)
            hi_b = round(bpms_m[int(0.90 * (n_b - 1))], 1)
            mix_bpm_bounds[mid] = (lo_b, hi_b)
        else:
            mix_bpm_bounds[mid] = (None, None)

    sft_counts: Counter[str] = Counter()
    train_dj_kept = 0
    dpo_count = 0

    try:
        for rec in clean_records:
            assert rec.split is not None
            rng = random.Random(f"{seed}:{rec.tran_id}")
            order = mix_track_order.get(rec.mix_id, [])
            pair_positions = [
                idx
                for idx, tid in enumerate(order)
                if tid == rec.prev_track.track_id
                and idx + 1 < len(order)
                and order[idx + 1] == rec.next_track.track_id
            ]
            if not pair_positions:
                pair_positions = [
                    idx for idx, tid in enumerate(order) if tid == rec.prev_track.track_id
                ]
            if pair_positions:
                pos = min(pair_positions, key=lambda idx: abs(idx - rec.i_tran))
                hist_ids = order[max(0, pos - history_window) : pos]
            else:
                pos = rec.i_tran
                hist_ids = []
            history = [_to_candidate(tracks[tid]) for tid in hist_ids if tid in tracks]

            mix = mixes[rec.mix_id]
            total_positions = max(mix.num_total_tracks - 1, len(order) - 1, rec.i_tran + 1, 1)
            set_pos = rec.i_tran if rec.i_tran >= 0 else pos
            progress = round(min(1.0, max(0.0, set_pos / float(total_positions))), 4)
            bpm_lo, bpm_hi = mix_bpm_bounds.get(rec.mix_id, (None, None))

            # Conditioning dropout on train: keep style_reference_dj ~50% of the time so the
            # model supports both DJ-steered ("mix like Four Tet") and DJ-free set briefs.
            if rec.split == "train":
                keep_dj = rng.random() >= dj_dropout_prob
                if keep_dj:
                    train_dj_kept += 1
                style_ref_dj = rec.dj_name if keep_dj else None
            else:
                style_ref_dj = rec.dj_name

            set_brief = SetBrief(
                genres=mix.genres[:5],
                curated_series=mix.curated_series,
                target_bpm_min=bpm_lo,
                target_bpm_max=bpm_hi,
                set_progress=progress,
                set_phase=classify_set_phase(progress),
                style_reference_dj=style_ref_dj,
            )

            exclude = {rec.prev_track.track_id, rec.next_track.track_id, *hist_ids}
            anchor_bpm = rec.prev_track.bpm or 125.0
            # Only bias distractors toward genre-tagged tracks when the target itself has
            # train-observed genres; otherwise target vs. distractor genre availability
            # becomes asymmetric on unseen val/test tracks.
            distractor_genres = set_brief.genres if rec.next_track.genres else []
            distractors = sample_hard_distractors(
                anchor_bpm, distractor_genres, exclude, candidate_pool_size - 1, rng
            )
            candidates = [_to_candidate(rec.next_track)] + [_to_candidate(d) for d in distractors]
            rng.shuffle(candidates)

            sft_ex = SFTExample(
                example_id=rec.tran_id,
                split=rec.split,
                tier=rec.quality.tier,
                composite_score=rec.quality.composite_score,
                mix_id=rec.mix_id,
                dj_name=rec.dj_name,
                set_brief=set_brief,
                history=history,
                current_track=_to_candidate(rec.prev_track),
                candidates=candidates,
                target_track_id=rec.next_track.track_id,
                target_plan=rec.plan,
            )
            sft_handles[rec.split].write(sft_ex.model_dump_json() + "\n")
            sft_counts[rec.split] += 1

            # Generate DPO pairs on gold + silver training transitions
            if rec.split == "train" and rec.quality.tier in ("gold", "silver") and distractors:
                # Pair A: BPM-matched wrong track (taste preference)
                neg_track = distractors[0]
                dpo_taste = DPOPair(
                    pair_id=f"{rec.tran_id}:taste",
                    split="train",
                    negative_type="bpm_matched_distractor",
                    mix_id=rec.mix_id,
                    set_brief=set_brief,
                    history=history,
                    current_track=_to_candidate(rec.prev_track),
                    candidates=candidates,
                    chosen_track_id=rec.next_track.track_id,
                    chosen_plan=rec.plan,
                    rejected_track_id=neg_track.track_id,
                    rejected_plan=rec.plan,
                )
                dpo_handle.write(dpo_taste.model_dump_json() + "\n")
                dpo_count += 1

                # Pair B: Right track, off-phrase / mistimed execution plan
                dpo_exec = DPOPair(
                    pair_id=f"{rec.tran_id}:exec",
                    split="train",
                    negative_type="off_phrase_plan",
                    mix_id=rec.mix_id,
                    set_brief=set_brief,
                    history=history,
                    current_track=_to_candidate(rec.prev_track),
                    candidates=candidates,
                    chosen_track_id=rec.next_track.track_id,
                    chosen_plan=rec.plan,
                    rejected_track_id=rec.next_track.track_id,
                    rejected_plan=_degrade_plan_off_phrase(rec.plan, rng),
                )
                dpo_handle.write(dpo_exec.model_dump_json() + "\n")
                dpo_count += 1
    finally:
        for fh in sft_handles.values():
            fh.close()
        dpo_handle.close()

    # Build summary stats for report
    failure_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    genre_counts: Counter[str] = Counter()
    series_counts: Counter[str] = Counter()
    split_djs: dict[str, set[str]] = defaultdict(set)

    for rec in records:
        tier_counts[rec.quality.tier] += 1
        for reason in rec.quality.failure_reasons:
            failure_counts[reason] += 1
        if rec.quality.passed_hard_gates and rec.split:
            style_counts[rec.plan.transition_style] += 1
            split_djs[rec.split].add(rec.dj_name_norm)
            for genre in mixes[rec.mix_id].genres:
                genre_counts[genre] += 1
            if rec.curated_series:
                series_counts[rec.curated_series] += 1

    # Verify zero DJ leakage across splits
    dj_overlap_train_val = len(split_djs["train"] & split_djs["val"])
    dj_overlap_train_test = len(split_djs["train"] & split_djs["test"])
    dj_overlap_val_test = len(split_djs["val"] & split_djs["test"])

    bpms = [r.prev_track.bpm for r in clean_records if r.prev_track.bpm is not None]
    overlaps = [r.plan.overlap_beats for r in clean_records]
    stretches = [abs(r.plan.tempo_stretch_next_pct) for r in clean_records]

    def _quantiles(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {}
        s = sorted(vals)
        n = len(s)
        return {
            "p10": round(s[int(0.10 * (n - 1))], 2),
            "p50": round(s[int(0.50 * (n - 1))], 2),
            "p90": round(s[int(0.90 * (n - 1))], 2),
            "mean": round(sum(s) / n, 2),
        }

    top_gold = [
        {
            "tran_id": r.tran_id,
            "score": r.quality.composite_score,
            "dj": r.dj_name,
            "series": r.curated_series,
            "prev": f"{r.prev_track.artist} - {r.prev_track.title} ({r.prev_track.bpm} BPM)",
            "next": f"{r.next_track.artist} - {r.next_track.title} ({r.next_track.bpm} BPM)",
            "overlap_beats": r.plan.overlap_beats,
            "style": r.plan.transition_style,
            "match_rates": [r.match_rate_prev, r.match_rate_next],
        }
        for r in clean_records[:10]
    ]

    summary = {
        "total_mixes_loaded": len(mixes),
        "total_tracks_indexed": len(tracks),
        "total_raw_transitions": len(records),
        "total_clean_transitions": len(clean_records),
        "retention_rate_pct": round(100.0 * len(clean_records) / max(1, len(records)), 2),
        "tier_counts": dict(tier_counts),
        "failure_reason_counts": dict(failure_counts.most_common()),
        "sft_split_counts": dict(sft_counts),
        "train_dj_conditioned_pct": round(
            100.0 * train_dj_kept / max(1, sft_counts["train"]), 2
        ),
        "dpo_train_pairs": dpo_count,
        "unique_djs_per_split": {k: len(v) for k, v in split_djs.items()},
        "dj_leakage_check": {
            "train_val_overlap": dj_overlap_train_val,
            "train_test_overlap": dj_overlap_train_test,
            "val_test_overlap": dj_overlap_val_test,
        },
        "transition_style_counts": dict(style_counts.most_common()),
        "top_genres": dict(genre_counts.most_common(15)),
        "curated_series_counts": dict(series_counts.most_common(10)),
        "distributions": {
            "bpm": _quantiles(bpms),
            "overlap_beats": _quantiles(overlaps),
            "abs_tempo_stretch_pct": _quantiles(stretches),
        },
        "top_10_gold_transitions": top_gold,
    }

    (out_dir / "data_quality_report.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
