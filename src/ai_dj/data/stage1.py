"""Stage 1 data ingestion, audio quality verification, and organization pipeline.

Builds a clean, verified `data/stage1/` hierarchy:
  - `house_pool/`: Filtered 116-136 BPM club house/tech-house/deep-house CC tracks
    analyzed with the <0.05 BPM hop=256 analyzer, quality-gated, and split track-disjoint.
  - `djmix_house/`: Filtered gold+silver MixesDB house transitions with verified
    DJ-disjoint train/val/test splits + Archive.org direct MP3 house set index.
  - `benchmark/`: Held-out commercial house reference tracks & pairs (never in train).
  - `verification_report.json`: Complete quality, distribution, and zero-leakage report.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ai_dj.audio.analyzer import AudioTrackCard, analyze_track
from ai_dj.audio.cc_pool import MTG_JAMENDO_GENRE_TSV_URL

HOUSE_STRICT_TAGS = frozenset({"house", "deephouse", "techhouse"})
CLUB_4X4_TAGS = frozenset({"house", "deephouse", "techhouse", "club", "techno", "dance"})
EXCLUDED_4X4_TAGS = frozenset(
    {
        "ambient",
        "chillout",
        "soundtrack",
        "drumnbass",
        "dubstep",
        "hiphop",
        "rap",
        "rock",
        "metal",
        "classical",
        "jazz",
        "folk",
        "acoustic",
        "orchestral",
        "trance",
    }
)

HOUSE_MIX_KEYWORDS = (
    "house",
    "tech house",
    "deep house",
    "progressive house",
    "electro house",
    "minimal house",
    "disco house",
)


def _load_diverse_house_candidates(
    tsv_path: Path,
    *,
    max_per_artist: int = 2,
    strict_only: bool = True,
) -> list[dict[str, Any]]:
    """Parse MTG-Jamendo genre TSV into artist-diverse house/deephouse/techhouse candidates."""
    if not tsv_path.exists():
        with urllib.request.urlopen(MTG_JAMENDO_GENRE_TSV_URL, timeout=30) as resp:
            tsv_path.parent.mkdir(parents=True, exist_ok=True)
            tsv_path.write_text(resp.read().decode("utf-8"), encoding="utf-8")

    per_artist: Counter[str] = Counter()
    candidates: list[dict[str, Any]] = []
    allowed = HOUSE_STRICT_TAGS if strict_only else CLUB_4X4_TAGS
    for line in tsv_path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 6:
            continue
        tid = parts[0].replace("track_", "").lstrip("0")
        artist_id = parts[1].replace("artist_", "").lstrip("0")
        album_id = parts[2].replace("album_", "").lstrip("0")
        dur_sec = float(parts[4]) if parts[4].replace(".", "", 1).isdigit() else 0.0
        if dur_sec < 120.0 or dur_sec > 540.0:
            continue
        genres = sorted(t.replace("genre---", "") for t in parts[5:] if t.startswith("genre---"))
        gset = set(genres)
        if not (gset & allowed) or (gset & EXCLUDED_4X4_TAGS):
            continue
        if per_artist[artist_id] >= max_per_artist:
            continue
        per_artist[artist_id] += 1
        candidates.append(
            {
                "track_id": tid,
                "artist_id": artist_id,
                "album_id": album_id,
                "expected_duration_sec": dur_sec,
                "genres": genres,
                "is_strict_house": bool(gset & HOUSE_STRICT_TAGS),
            }
        )
    # Deterministic shuffle via sha256(track_id) so we sample across the whole catalog
    candidates.sort(key=lambda c: hashlib.sha256(c["track_id"].encode("utf-8")).hexdigest())
    return candidates


def _download_one_jamendo(
    cand: dict[str, Any],
    raw_dir: Path,
    min_bytes: int,
    max_bytes: int,
) -> dict[str, Any] | None:
    tid = cand["track_id"]
    out_file = raw_dir / f"{tid}.mp3"
    if out_file.exists() and min_bytes <= out_file.stat().st_size <= max_bytes:
        nbytes = int(out_file.stat().st_size)
    else:
        url = f"https://mp3l.jamendo.com/?trackid={tid}&format=mp31"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ai-dj-stage1/0.1"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
        except Exception:  # noqa: BLE001
            return None
        nbytes = len(data)
        if not (min_bytes <= nbytes <= max_bytes):
            return None
        out_file.write_bytes(data)

    return {
        "track_id": tid,
        "artist_id": cand["artist_id"],
        "album_id": cand["album_id"],
        "filename": out_file.name,
        "audio_path": str(out_file),
        "bytes": nbytes,
        "genres": cand["genres"],
        "is_strict_house": cand["is_strict_house"],
        "source": "MTG-Jamendo-CC",
        "source_url": f"https://www.jamendo.com/track/{tid}",
    }


def ingest_stage1_house_pool(
    out_dir: Path,
    *,
    target_tracks: int = 100,
    max_per_artist: int = 2,
    min_bytes: int = 1_200_000,
    max_bytes: int = 8_500_000,
    workers: int = 6,
) -> Path:
    """Download artist-diverse strict house/deephouse/techhouse CC tracks into `out_dir`."""
    out_dir = out_dir.resolve()
    raw_dir = out_dir / "raw_mp3"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    tsv_cache = out_dir / "mtg_jamendo_genre.tsv"
    if not tsv_cache.exists():
        existing_tsv = Path("data/raw/cc_electronic/mtg_jamendo_genre.tsv")
        if existing_tsv.exists():
            tsv_cache.write_text(existing_tsv.read_text(encoding="utf-8"), encoding="utf-8")

    candidates = _load_diverse_house_candidates(
        tsv_cache, max_per_artist=max_per_artist, strict_only=True
    )
    pool_slice = candidates[: int(target_tracks * 1.4)]

    manifest: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(
            lambda c: _download_one_jamendo(c, raw_dir, min_bytes, max_bytes), pool_slice
        ):
            if res is not None:
                manifest.append(res)
                if len(manifest) >= target_tracks:
                    break

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def _analyze_one_item(
    item: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    """Worker function for parallel MIR analysis."""
    audio_path = Path(item["audio_path"])
    tid = str(item["track_id"])
    try:
        card = analyze_track(audio_path, track_id=tid)
        card.genres = list(item.get("genres") or ["house"])
        return item, card.model_dump(), None
    except Exception as exc:  # noqa: BLE001
        return item, None, f"decode_error: {type(exc).__name__}: {exc}"


def verify_and_split_house_pool(
    manifest_path: Path,
    out_dir: Path,
    *,
    bpm_min: float = 116.0,
    bpm_max: float = 136.0,
    min_stability: float = 0.85,
    min_duration_sec: float = 90.0,
    min_lufs: float = -30.0,
    min_peak_dbfs: float = -6.0,
    min_sections: int = 4,
    val_frac: float = 0.20,
    workers: int | None = None,
) -> dict[str, Any]:
    """Analyze tracks with the <0.05 BPM analyzer, apply quality gates, and split track-disjoint."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    max_workers = workers or max(1, min(8, (os.cpu_count() or 2)))
    verified_cards: list[AudioTrackCard] = []
    rejected: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_analyze_one_item, manifest))

    for item, card_dict, err in results:
        tid = str(item["track_id"])
        if err is not None or card_dict is None:
            rejection_counts["decode_error"] += 1
            rejected.append({"track_id": tid, "reason": err or "decode_error"})
            continue
        card = AudioTrackCard.model_validate(card_dict)
        reasons: list[str] = []
        if card.duration_sec < min_duration_sec:
            reasons.append(f"short_duration({card.duration_sec:.1f}s)")
        if not (bpm_min <= card.bpm <= bpm_max):
            reasons.append(f"bpm_out_of_range({card.bpm:.2f})")
        if card.bpm_stability < min_stability:
            reasons.append(f"low_bpm_stability({card.bpm_stability:.3f})")
        if card.integrated_lufs < min_lufs:
            reasons.append(f"too_quiet({card.integrated_lufs:.1f}LUFS)")
        if card.peak_dbfs < min_peak_dbfs:
            reasons.append(f"low_peak({card.peak_dbfs:.1f}dBFS)")
        if len(card.sections) < min_sections:
            reasons.append(f"too_few_sections({len(card.sections)})")

        if reasons:
            primary = reasons[0].split("(")[0]
            rejection_counts[primary] += 1
            rejected.append(
                {
                    "track_id": tid,
                    "bpm": card.bpm,
                    "bpm_stability": card.bpm_stability,
                    "duration_sec": card.duration_sec,
                    "genres": card.genres,
                    "reasons": reasons,
                }
            )
            continue
        verified_cards.append(card)

    # Deterministic track-disjoint split (distinct salt from candidate sampling hash)
    verified_cards.sort(key=lambda c: c.track_id)
    train_cards: list[AudioTrackCard] = []
    val_cards: list[AudioTrackCard] = []
    for card in verified_cards:
        salted = f"stage1_split:{card.track_id}".encode()
        h = int(hashlib.sha256(salted).hexdigest()[:8], 16) / 0xFFFFFFFF
        if h < val_frac:
            val_cards.append(card)
        else:
            train_cards.append(card)

    # Assert zero track leakage
    train_ids = {c.track_id for c in train_cards}
    val_ids = {c.track_id for c in val_cards}
    assert train_ids.isdisjoint(val_ids), "Track leakage between train and val!"

    def _write_jsonl(path: Path, cards: list[AudioTrackCard]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for c in cards:
                f.write(c.model_dump_json() + "\n")

    _write_jsonl(out_dir / "all_verified_cards.jsonl", verified_cards)
    _write_jsonl(out_dir / "train_cards.jsonl", train_cards)
    _write_jsonl(out_dir / "val_cards.jsonl", val_cards)
    (out_dir / "rejected_tracks.json").write_text(json.dumps(rejected, indent=2), encoding="utf-8")

    bpms = [c.bpm for c in verified_cards]
    stabs = [c.bpm_stability for c in verified_cards]
    durs = [c.duration_sec for c in verified_cards]
    keys = Counter(c.camelot for c in verified_cards)
    genres = Counter(g for c in verified_cards for g in c.genres)
    return {
        "raw_candidates": len(manifest),
        "verified_total": len(verified_cards),
        "train_tracks": len(train_cards),
        "val_tracks": len(val_cards),
        "rejected_total": len(rejected),
        "rejection_breakdown": dict(rejection_counts),
        "track_leakage_count": len(train_ids & val_ids),
        "bpm_min": min(bpms) if bpms else None,
        "bpm_max": max(bpms) if bpms else None,
        "bpm_mean": round(sum(bpms) / len(bpms), 2) if bpms else None,
        "stability_mean": round(sum(stabs) / len(stabs), 4) if stabs else None,
        "duration_mean_sec": round(sum(durs) / len(durs), 1) if durs else None,
        "camelot_distribution": dict(keys.most_common()),
        "genre_distribution": dict(genres.most_common(12)),
    }


def organize_djmix_house_corpus(
    transitions_clean_path: Path,
    djmix_raw_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Filter enriched MixesDB transitions to gold+silver house and verify DJ-disjoint splits."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    djs_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    tier_counts: Counter[str] = Counter()
    style_counts: Counter[str] = Counter()
    total_scanned = 0
    total_house_kept = 0

    with transitions_clean_path.open("r", encoding="utf-8") as f:
        for line in f:
            total_scanned += 1
            rec = json.loads(line)
            tags = " ".join(rec.get("mix_tags") or []).lower()
            if not any(kw in tags for kw in HOUSE_MIX_KEYWORDS):
                continue
            tier = (rec.get("quality") or {}).get("tier", "bronze")
            if tier not in ("gold", "silver"):
                continue
            plan = rec.get("plan") or {}
            ov = float(plan.get("overlap_beats") or 0.0)
            if not (8.0 <= ov <= 128.0):
                continue
            bpm_delta = rec.get("bpm_delta")
            if bpm_delta is not None and abs(float(bpm_delta)) > 8.0:
                continue

            sp = str(rec.get("split") or "train")
            if sp not in splits:
                sp = "train"
            splits[sp].append(rec)
            dj_name = str(
                rec.get("dj_name_norm") or rec.get("dj_name") or rec.get("mix_id") or "unknown"
            ).lower().strip()
            djs_by_split[sp].add(dj_name)
            tier_counts[tier] += 1
            style_counts[str(plan.get("transition_style") or "unknown")] += 1
            total_house_kept += 1

    # Verify DJ-disjoint splits
    tv_leak = djs_by_split["train"] & djs_by_split["val"]
    tt_leak = djs_by_split["train"] & djs_by_split["test"]
    vt_leak = djs_by_split["val"] & djs_by_split["test"]

    for sp, rows in splits.items():
        p = out_dir / f"house_transitions_{sp}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # Index direct Archive.org MP3 house sets for Stage 2 calibration / hillclimbing
    raw = json.loads(djmix_raw_path.read_text(encoding="utf-8"))
    mixes = raw if isinstance(raw, list) else list(raw.values())
    archive_house: list[dict[str, Any]] = []
    for m in mixes:
        if not isinstance(m, dict):
            continue
        url = str(m.get("audio_url") or "")
        tags = json.dumps(m.get("tags") or []).lower()
        if ("archive.org" in url or url.endswith(".mp3")) and any(
            kw in tags for kw in HOUSE_MIX_KEYWORDS
        ):
            archive_house.append(
                {
                    "mix_id": m.get("id"),
                    "title": m.get("title"),
                    "tags": m.get("tags"),
                    "audio_url": url,
                    "num_available_transitions": int(m.get("num_available_transitions") or 0),
                }
            )
    archive_house.sort(key=lambda x: -x["num_available_transitions"])
    with (out_dir / "archive_org_house_sets.jsonl").open("w", encoding="utf-8") as f:
        for item in archive_house:
            f.write(json.dumps(item) + "\n")

    return {
        "transitions_scanned": total_scanned,
        "house_gold_silver_kept": total_house_kept,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "unique_djs_by_split": {k: len(v) for k, v in djs_by_split.items()},
        "dj_leakage": {
            "train_val": len(tv_leak),
            "train_test": len(tt_leak),
            "val_test": len(vt_leak),
        },
        "tier_counts": dict(tier_counts),
        "style_counts": dict(style_counts),
        "archive_org_house_sets": len(archive_house),
        "archive_org_house_transitions": sum(x["num_available_transitions"] for x in archive_house),
    }


def build_commercial_benchmark(
    tracks: list[tuple[str, Path, list[str]]],
    out_dir: Path,
) -> dict[str, Any]:
    """Analyze held-out commercial house reference tracks with the <0.05 BPM analyzer."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cards: list[AudioTrackCard] = []
    for tid, path, genres in tracks:
        if not path.exists():
            continue
        card = analyze_track(path, track_id=tid)
        card.genres = genres
        cards.append(card)

    with (out_dir / "benchmark_cards.jsonl").open("w", encoding="utf-8") as f:
        for c in cards:
            f.write(c.model_dump_json() + "\n")

    pairs = [
        {
            "pair_id": "nanana_to_thinking_about_you",
            "track_a": "peggy_gou_nanana",
            "track_b": "calvin_harris_thinking_about_you",
            "note": "Close-key club house pair (5A -> 4A, 130.01 -> 128.00 BPM)",
        },
        {
            "pair_id": "world_hold_on_to_nanana",
            "track_a": "world_hold_on_fisher",
            "track_b": "peggy_gou_nanana",
            "note": (
                "Distant-key tech-house pair (1A -> 5A, 127.02 -> 130.01 BPM, tests drum-bridge)"
            ),
        },
    ]
    (out_dir / "benchmark_pairs.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
    return {
        "benchmark_tracks": [
            {
                "track_id": c.track_id,
                "bpm": c.bpm,
                "camelot": c.camelot,
                "bpm_stability": c.bpm_stability,
                "duration_sec": c.duration_sec,
                "sections": len(c.sections),
            }
            for c in cards
        ],
        "benchmark_pairs": len(pairs),
    }


def _map_mixesdb_to_dsp_plan(rec: dict[str, Any]) -> dict[str, Any]:
    """Convert a real MixesDB house transition into our executable parametric mix schema."""
    plan = rec.get("plan") or {}
    prev_t = rec.get("prev_track") or {}
    next_t = rec.get("next_track") or {}
    bpm_delta_abs = abs(float(rec.get("bpm_delta") or 0.0))
    raw_ov = float(plan.get("overlap_beats") or 32.0)
    ov = int(max(8, min(64, round(raw_ov / 8.0) * 8)))

    n_beats_prev = max(32, int(prev_t.get("beat_count") or 512))
    n_beats_next = max(32, int(next_t.get("beat_count") or 512))
    raw_cue_out = int(round(float(plan.get("track_cue_out_beat_prev") or (n_beats_prev - ov - 8))))
    cue_out = max(16, min(n_beats_prev - ov - 4, raw_cue_out))
    raw_cue_in = int(round(float(plan.get("track_cue_in_beat_next") or 0)))
    cue_in = max(0, min(max(0, n_beats_next - ov - 4), raw_cue_in))

    raw_style = str(plan.get("transition_style") or "standard_blend")
    tid_str = str(rec.get("tran_id") or "")
    tid_bit = int(hashlib.sha256(f"style:{tid_str}".encode()).hexdigest()[:8], 16) % 2
    if bpm_delta_abs > 5.0 or ov <= 8:
        style = "quick_cut"
        ov = 8
        auto = (0.50, 0.35, 0.35, 0.40, 0.50)
    elif raw_style in ("long_blend", "extended_overlay") or ov >= 48:
        style = "long_blend"
        auto = (0.50, 0.50, 0.72, 0.20, 0.58)
    elif raw_style == "short_blend" or ov <= 24:
        if bpm_delta_abs >= 2.5:
            style = "filter_sweep"
            auto = (0.50, 0.40, 0.52, 0.32, 0.58)
        else:
            style = "bass_swap"
            auto = (0.50, 0.45, 0.62, 0.25, 0.55)
    else:
        if bpm_delta_abs >= 1.5 or tid_bit == 0:
            style = "stem_strip_blend"
            auto = (0.50, 0.42, 0.45, 0.48, 0.60)
        else:
            style = "bass_swap"
            auto = (0.50, 0.45, 0.62, 0.25, 0.55)

    return {
        "next_track_id": str(next_t.get("track_id") or "next_track"),
        "transition_style": style,
        "overlap_beats": ov,
        "cue_out_beat_prev": cue_out,
        "cue_in_beat_next": cue_in,
        "bass_swap_beat_frac": auto[0],
        "hat_morph_end_frac": auto[1],
        "harmony_strip_a_end_frac": auto[2],
        "harmony_entry_b_start_frac": auto[3],
        "vocal_entry_b_start_frac": auto[4],
    }


def export_stage1_house_sft_chat(
    djmix_house_dir: Path,
    out_dir: Path,
    *,
    dj_dropout: float = 0.30,
) -> dict[str, Any]:
    """Export all MixesDB house train/val/test transitions to Inkling-Small SFT chat JSONL."""
    from ai_dj.rl.audio_native import AUDIO_NATIVE_SYSTEM_PROMPT

    out_dir.mkdir(parents=True, exist_ok=True)
    split_stats: dict[str, Any] = {}

    for sp in ("train", "val", "test"):
        src = djmix_house_dir / f"house_transitions_{sp}.jsonl"
        if not src.exists():
            continue
        rows: list[dict[str, Any]] = []
        style_counts: Counter[str] = Counter()
        with src.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                prev_t = rec.get("prev_track") or {}
                next_t = rec.get("next_track") or {}
                target = _map_mixesdb_to_dsp_plan(rec)
                style_counts[target["transition_style"]] += 1

                tid = str(rec.get("tran_id") or "")
                h = int(hashlib.sha256(f"dj_drop:{tid}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
                dj_line = (
                    f"Reference DJ: {rec.get('dj_name')} "
                    f"({', '.join((rec.get('mix_tags') or [])[:4])})\n"
                    if h >= dj_dropout and rec.get("dj_name")
                    else "Reference DJ: generic club house DJ\n"
                )
                prev_genres = ", ".join((prev_t.get("genres") or ["House"])[:4])
                next_genres = ", ".join((next_t.get("genres") or ["House"])[:4])
                user_prompt = (
                    f"{dj_line}"
                    "Plan the real-DJ house transition from Deck A to Deck B.\n\n"
                    f"DECK A (outgoing): {prev_t.get('track_id')} | "
                    f"{prev_t.get('artist')} - {prev_t.get('title')}\n"
                    f"  bpm={float(prev_t.get('bpm') or 125.0):.2f} "
                    f"stability={float(prev_t.get('bpm_stability') or 0.95):.3f} "
                    f"duration={float(prev_t.get('duration_sec') or 300.0):.1f}s "
                    f"total_beats={int(prev_t.get('beat_count') or 512)} "
                    f"genres=[{prev_genres}]\n"
                    f"DECK B (incoming): {next_t.get('track_id')} | "
                    f"{next_t.get('artist')} - {next_t.get('title')}\n"
                    f"  bpm={float(next_t.get('bpm') or 125.0):.2f} "
                    f"stability={float(next_t.get('bpm_stability') or 0.95):.3f} "
                    f"duration={float(next_t.get('duration_sec') or 300.0):.1f}s "
                    f"total_beats={int(next_t.get('beat_count') or 512)} "
                    f"genres=[{next_genres}]\n"
                    f"BPM delta: {float(rec.get('bpm_delta') or 0.0):+.2f} BPM\n\n"
                    "Return ONLY the single valid JSON mix plan."
                )
                rows.append(
                    {
                        "tran_id": tid,
                        "split": sp,
                        "tier": (rec.get("quality") or {}).get("tier", "silver"),
                        "dj_name": rec.get("dj_name"),
                        "messages": [
                            {"role": "system", "content": AUDIO_NATIVE_SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": json.dumps(target)},
                        ],
                    }
                )
        out_path = out_dir / f"house_sft_{sp}_chat.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        split_stats[sp] = {
            "path": str(out_path),
            "examples": len(rows),
            "style_counts": dict(style_counts),
        }

    summary = {"splits": split_stats}
    (out_dir / "house_sft_export_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
