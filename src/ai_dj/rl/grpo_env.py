"""Audio-grounded GRPO task builder, prompt formatter, and DSP-verified reward engine."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from ai_dj.audio.analyzer import (
    AudioTrackCard,
    TrackSection,
    camelot_distance,
    timbre_cosine_similarity,
)
from ai_dj.audio.renderer import RenderSpec, render_transition
from ai_dj.audio.verifier import verify_transition
from ai_dj.training.prompts import _extract_json_object

CreativeStyle = Literal[
    "bass_swap",
    "filter_sweep",
    "stem_strip_blend",
    "quick_cut",
    "long_blend",
]
CREATIVE_STYLES: tuple[CreativeStyle, ...] = (
    "bass_swap",
    "filter_sweep",
    "stem_strip_blend",
    "quick_cut",
    "long_blend",
)

GRPO_SYSTEM_PROMPT = (
    "You are an expert electronic club DJ planner with full MIR audio analysis of every track.\n"
    "Given Deck A and an audio-analyzed candidate crate (BPM, stretch %, Camelot key & distance, "
    "timbre similarity, genres, vocal density, energy, and recommended intro/outro cue beats), "
    "choose the best next track and a creative DSP transition plan.\n"
    "Choose the technique that fits the audio:\n"
    '- "bass_swap": tight 4/4 beatmatch (<=4% stretch) with low vocals; swap bass cleanly.\n'
    '- "filter_sweep": smooth HP/LP filter sweep across energy contrasts or longer blends.\n'
    '- "stem_strip_blend": strip outgoing melody via HPSS when vocals or dense mids overlap.\n'
    '- "long_blend": 48-64 beat hypnotic blend when BPM/key/timbre match and vocals are low.\n'
    '- "quick_cut": short 8-beat handoff when BPM gap >5% or vocal clash is unavoidable.\n'
    "Respond with ONLY a single valid JSON object with keys in order: "
    '"next_track_id", "transition_style", "overlap_beats", '
    '"cue_out_beat_prev", "cue_in_beat_next", "bass_swap_beat_frac".'
)


@dataclass(frozen=True)
class AudioTransitionTask:
    """One audio-grounded DJ transition decision over verified electronic tracks."""

    task_id: str
    deck_a: AudioTrackCard
    candidates: list[AudioTrackCard]
    set_phase: str = "peak"
    target_genres: tuple[str, ...] = ("electronic", "house", "trance")


@dataclass(frozen=True)
class CreativeMixPlan:
    """Parsed creative transition plan emitted by the RL policy."""

    next_track_id: str
    transition_style: CreativeStyle
    overlap_beats: int
    cue_out_beat_prev: int
    cue_in_beat_next: int
    bass_swap_beat_frac: float = 0.5
    hat_morph_end_frac: float | None = None
    harmony_strip_a_end_frac: float | None = None
    harmony_entry_b_start_frac: float | None = None
    vocal_entry_b_start_frac: float | None = None


@dataclass(frozen=True)
class GRPORewardBreakdown:
    """Detailed verifiable reward for a single rollout."""

    total_reward: float
    schema_valid: bool
    compatibility_reward: float
    dsp_qa_reward: float
    craft_reward: float
    plan: CreativeMixPlan | None = None
    qa_flags: tuple[str, ...] = ()
    error: str | None = None


def load_electronic_track_cards(
    cards_jsonl: Path,
    *,
    min_local_stability: float = 0.70,
    min_bpm: float = 105.0,
    max_bpm: float = 155.0,
) -> list[AudioTrackCard]:
    """Load club-stable electronic AudioTrackCards from JSONL."""
    cards: list[AudioTrackCard] = []
    with cards_jsonl.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            card = AudioTrackCard.model_validate_json(line)
            if (
                card.local_bpm_stability >= min_local_stability
                and min_bpm <= card.bpm <= max_bpm
                and len(card.beat_times) >= 64
                and Path(card.audio_path).exists()
            ):
                cards.append(card)
    return cards


def build_audio_grpo_tasks(
    cards: list[AudioTrackCard],
    *,
    crate_size: int = 8,
    num_tasks: int = 40,
    seed: int = 42,
) -> list[AudioTransitionTask]:
    """Construct diverse audio-grounded DJ transition tasks from verified electronic cards."""
    if len(cards) < 3:
        raise ValueError(f"Need at least 3 club-stable electronic tracks, got {len(cards)}.")

    rng = random.Random(seed)
    phases = ("opener", "build", "peak", "closer")
    tasks: list[AudioTransitionTask] = []
    effective_crate = min(crate_size, len(cards) - 1)

    for idx in range(num_tasks):
        deck_a = cards[idx % len(cards)]
        others = [c for c in cards if c.track_id != deck_a.track_id]
        # Score others by BPM+Camelot+timbre so every crate has both strong matches & distractors
        others_sorted = sorted(
            others,
            key=lambda c: (
                abs(c.bpm - deck_a.bpm) / max(deck_a.bpm, 1.0)
                + 0.03 * camelot_distance(deck_a.camelot, c.camelot)
                - 0.10 * timbre_cosine_similarity(deck_a, c)
            ),
        )
        n_good = max(2, effective_crate // 2)
        good_pool = others_sorted[: min(len(others_sorted), n_good + 2)]
        rest_pool = others_sorted[min(len(others_sorted), n_good + 2) :]
        chosen = rng.sample(good_pool, min(n_good, len(good_pool)))
        needed = effective_crate - len(chosen)
        if needed > 0 and rest_pool:
            chosen.extend(rng.sample(rest_pool, min(needed, len(rest_pool))))
        # Fill if pool is small
        for c in others:
            if len(chosen) >= effective_crate:
                break
            if c.track_id not in {x.track_id for x in chosen}:
                chosen.append(c)
        rng.shuffle(chosen)

        phase = phases[idx % len(phases)]
        genres = tuple(deck_a.genres[:3]) if deck_a.genres else ("electronic", "house")
        tasks.append(
            AudioTransitionTask(
                task_id=f"audio-grpo-{idx:03d}",
                deck_a=deck_a,
                candidates=chosen,
                set_phase=phase,
                target_genres=genres,
            )
        )
    return tasks


def format_section_timeline(card: AudioTrackCard, *, max_sections: int = 12) -> str:
    """Format a track's phrase-aligned structural sections into a compact prompt timeline."""
    if not card.sections:
        return ""
    secs = card.sections
    if len(secs) > max_sections:
        step = max(1, len(secs) // max_sections)
        secs = secs[::step][: max_sections - 1] + [secs[-1]]
    parts = [
        (
            f"S{s.index}:{s.label}@b{s.start_beat}-{s.end_beat}"
            f"(E={s.energy:.2f},bass={s.bass_density:.2f},voc={s.vocal_ratio:.2f})"
        )
        for s in secs
    ]
    return " | ".join(parts)


def build_pair_mix_tasks(
    cards: list[AudioTrackCard],
    *,
    num_tasks: int = 40,
    seed: int = 42,
    max_bpm_stretch_pct: float = 5.0,
) -> list[AudioTransitionTask]:
    """Construct 2-song full-track pair mixing tasks (Deck A + fixed Deck B)."""
    if len(cards) < 2:
        raise ValueError(f"Need at least 2 club-stable electronic tracks, got {len(cards)}.")
    rng = random.Random(seed)
    pairs: list[tuple[AudioTrackCard, AudioTrackCard]] = []
    for a in cards:
        for b in cards:
            if a.track_id == b.track_id:
                continue
            stretch = abs((a.bpm - b.bpm) / max(b.bpm, 1.0)) * 100.0
            if stretch <= max_bpm_stretch_pct:
                pairs.append((a, b))
    if not pairs:
        pairs = [(a, b) for a in cards for b in cards if a.track_id != b.track_id]
    rng.shuffle(pairs)
    phases = ("opener", "build", "peak", "closer")
    tasks: list[AudioTransitionTask] = []
    for idx in range(num_tasks):
        a, b = pairs[idx % len(pairs)]
        genres = tuple(a.genres[:3]) if a.genres else ("electronic", "house")
        tasks.append(
            AudioTransitionTask(
                task_id=f"pair-mix-{idx:03d}",
                deck_a=a,
                candidates=[b],
                set_phase=phases[idx % len(phases)],
                target_genres=genres,
            )
        )
    return tasks


def format_audio_grpo_prompt(task: AudioTransitionTask) -> str:
    """Format an audio-grounded DJ transition prompt with MIR + section timeline features."""
    a = task.deck_a
    a_outro = (
        a.recommended_outro_beat
        if a.recommended_outro_beat is not None
        else max(32, len(a.beat_times) - 64)
    )
    lines: list[str] = [
        "### SetBrief",
        f"- Genres: {', '.join(task.target_genres)}",
        f"- Target BPM: {a.bpm * 0.94:.1f}–{a.bpm * 1.06:.1f}",
        f"- Set Phase: {task.set_phase}",
        "",
        "### Current Track (Deck A)",
        (
            f"  id={a.track_id} | genres={','.join(a.genres[:3]) or 'electronic'} | "
            f"BPM={a.bpm:.1f} (stab={a.local_bpm_stability:.2f}) | key={a.camelot} | "
            f"beats={len(a.beat_times)} | rec_outro_beat={a_outro} | "
            f"vocal_ratio={a.vocal_ratio:.2f} | energy={a.mean_energy:.2f}"
        ),
    ]
    a_tl = format_section_timeline(a, max_sections=12)
    if a_tl:
        lines.append(f"  Deck A sections: {a_tl}")
    lines.append("")
    lines.append(f"### Candidate Crate ({len(task.candidates)} audio-analyzed tracks)")
    max_cand_secs = 12 if len(task.candidates) == 1 else 6
    for i, c in enumerate(task.candidates, start=1):
        stretch_pct = ((a.bpm - c.bpm) / max(c.bpm, 1.0)) * 100.0
        cdist = camelot_distance(a.camelot, c.camelot)
        tsim = timbre_cosine_similarity(a, c)
        c_intro = c.recommended_intro_beat if c.recommended_intro_beat is not None else 16
        lines.append(
            f"  {i:02d}. id={c.track_id} | genres={','.join(c.genres[:3]) or 'electronic'} | "
            f"BPM={c.bpm:.1f} (stretch={stretch_pct:+.1f}%) | key={c.camelot} (dist={cdist}) | "
            f"timbre_sim={tsim:.2f} | beats={len(c.beat_times)} | rec_intro_beat={c_intro} | "
            f"vocal_ratio={c.vocal_ratio:.2f} | energy={c.mean_energy:.2f}"
        )
        c_tl = format_section_timeline(c, max_sections=max_cand_secs)
        if c_tl:
            lines.append(f"      sections: {c_tl}")
    lines.append("")
    lines.append(
        "Select the best next track, exact phrase-aligned cue moments from the song sections, "
        "and creative transition style as JSON."
    )
    return "\n".join(lines)


def parse_creative_plan(
    raw_text: str, valid_candidate_ids: set[str]
) -> tuple[CreativeMixPlan | None, str | None]:
    """Parse and validate a creative transition plan JSON completion."""
    obj = _extract_json_object(raw_text)
    if obj is None:
        return None, "invalid_json"

    tid = str(obj.get("next_track_id", "")).strip()
    if tid not in valid_candidate_ids:
        return None, "invalid_track_id"

    style_raw = str(obj.get("transition_style", "")).strip()
    # Map legacy SFT styles gracefully if the SFT-v2 policy emits them during early exploration
    style_alias = {
        "standard_blend": "bass_swap",
        "short_blend": "filter_sweep",
        "extended_overlay": "long_blend",
    }
    style_mapped = style_alias.get(style_raw, style_raw)
    if style_mapped not in CREATIVE_STYLES:
        return None, f"invalid_style:{style_raw}"

    try:
        overlap = int(round(float(obj.get("overlap_beats", 32))))
        cue_out = int(round(float(obj.get("cue_out_beat_prev", 0))))
        cue_in = int(round(float(obj.get("cue_in_beat_next", 0))))
        bs_frac = float(obj.get("bass_swap_beat_frac", 0.5))
    except (TypeError, ValueError):
        return None, "non_numeric_fields"

    if not (4 <= overlap <= 96):
        return None, f"overlap_out_of_range:{overlap}"
    if cue_out < 0 or cue_in < 0:
        return None, "negative_cue"
    bs_frac = min(0.85, max(0.15, bs_frac))

    def _opt_frac(key: str, lo: float, hi: float) -> float | None:
        val = obj.get(key)
        if val is None:
            return None
        try:
            return round(min(hi, max(lo, float(val))), 2)
        except (TypeError, ValueError):
            return None

    return (
        CreativeMixPlan(
            next_track_id=tid,
            transition_style=style_mapped,  # type: ignore[arg-type]
            overlap_beats=overlap,
            cue_out_beat_prev=cue_out,
            cue_in_beat_next=cue_in,
            bass_swap_beat_frac=round(bs_frac, 2),
            hat_morph_end_frac=_opt_frac("hat_morph_end_frac", 0.15, 0.90),
            harmony_strip_a_end_frac=_opt_frac("harmony_strip_a_end_frac", 0.15, 0.95),
            harmony_entry_b_start_frac=_opt_frac("harmony_entry_b_start_frac", 0.05, 0.85),
            vocal_entry_b_start_frac=_opt_frac("vocal_entry_b_start_frac", 0.15, 0.90),
        ),
        None,
    )


def _section_at_beat(card: AudioTrackCard, beat: int) -> TrackSection | None:
    if not card.sections:
        return None
    last_idx = len(card.sections) - 1
    for idx, sec in enumerate(card.sections):
        if sec.start_beat <= beat < sec.end_beat or (
            idx == last_idx and sec.start_beat <= beat <= sec.end_beat
        ):
            return sec
    return card.sections[-1] if beat > card.sections[-1].end_beat else card.sections[0]


def _section_handoff_score(
    deck_a: AudioTrackCard,
    deck_b: AudioTrackCard,
    plan: CreativeMixPlan,
) -> float:
    """Score [0..1] whether chosen cue_out/cue_in land on musically strong structural moments."""
    sec_a = _section_at_beat(deck_a, plan.cue_out_beat_prev)
    sec_b = _section_at_beat(deck_b, plan.cue_in_beat_next)
    if sec_a is None or sec_b is None:
        return 0.75

    # Outgoing Deck A: outro, breakdown, and groove are strongest mix-out points
    out_scores = {
        "outro": 1.0,
        "breakdown": 0.95,
        "groove": 0.85,
        "drop": 0.65,
        "buildup": 0.45,
        "intro": 0.25,
    }
    # Incoming Deck B: intro, buildup, and clean groove are strongest mix-in points
    in_scores = {
        "intro": 1.0,
        "buildup": 0.90,
        "groove": 0.85,
        "breakdown": 0.75,
        "drop": 0.55,
        "outro": 0.20,
    }
    s_out = out_scores.get(sec_a.label, 0.7)
    s_in = in_scores.get(sec_b.label, 0.7)
    # Bonus for snapping close to a section boundary (within 8 beats)
    near_a = 1.0 if abs(plan.cue_out_beat_prev - sec_a.start_beat) <= 8 else 0.75
    near_b = 1.0 if abs(plan.cue_in_beat_next - sec_b.start_beat) <= 8 else 0.75
    # Penalize overlapping two high-vocal sections unless stripping stems or quick-cutting
    both_vocal = sec_a.vocal_ratio >= 0.45 and sec_b.vocal_ratio >= 0.45
    vocal_pen = (
        0.0
        if not both_vocal or plan.transition_style in ("stem_strip_blend", "quick_cut")
        else 0.25
    )
    score = 0.40 * s_out + 0.40 * s_in + 0.10 * near_a + 0.10 * near_b - vocal_pen
    return max(0.0, min(1.0, score))


def _creative_technique_fit(
    deck_a: AudioTrackCard,
    deck_b: AudioTrackCard,
    plan: CreativeMixPlan,
    stretch_pct_abs: float,
    cam_dist: int,
    timbre_sim: float,
) -> float:
    """Score [0..1] how well the chosen creative technique matches the chosen sections' audio."""
    sec_a = _section_at_beat(deck_a, plan.cue_out_beat_prev)
    sec_b = _section_at_beat(deck_b, plan.cue_in_beat_next)
    voc_a = sec_a.vocal_ratio if sec_a is not None else deck_a.vocal_ratio
    voc_b = sec_b.vocal_ratio if sec_b is not None else deck_b.vocal_ratio
    e_a = sec_a.energy if sec_a is not None else deck_a.mean_energy
    e_b = sec_b.energy if sec_b is not None else deck_b.mean_energy
    high_vocals = max(voc_a, voc_b) >= 0.35
    perc_a = sec_a.percussion_ratio if sec_a is not None else 0.5
    perc_b = sec_b.percussion_ratio if sec_b is not None else 0.5
    distinct_drums = timbre_sim < 0.88 or (perc_a >= 0.45 and perc_b >= 0.45)
    energy_gap = abs(e_a - e_b)
    style = plan.transition_style

    if style == "quick_cut":
        if stretch_pct_abs > 5.0 or cam_dist >= 4:
            return 1.0
        return 0.30 if plan.overlap_beats <= 16 else 0.15
    if style == "stem_strip_blend":
        if plan.overlap_beats < 24:
            return 0.30
        if (
            high_vocals
            or distinct_drums
            or (sec_a is not None and sec_a.label in ("drop", "groove"))
        ):
            return 1.0
        return 0.85
    if style == "filter_sweep":
        if plan.overlap_beats < 24:
            return 0.30
        if energy_gap >= 0.12 or cam_dist in (1, 2):
            return 1.0
        return 0.80
    if style == "bass_swap":
        if plan.overlap_beats < 24:
            return 0.25
        if stretch_pct_abs <= 4.5 and not high_vocals and plan.overlap_beats in (24, 32, 48):
            return 0.95 if not distinct_drums else 0.80
        return 0.65
    if style == "long_blend":
        if plan.overlap_beats < 32:
            return 0.25
        if (
            stretch_pct_abs <= 3.5
            and cam_dist <= 2
            and timbre_sim >= 0.80
            and not high_vocals
            and plan.overlap_beats >= 32
        ):
            return 1.0
        return 0.60
    return 0.5


def score_creative_plan(
    task: AudioTransitionTask,
    raw_text: str,
    *,
    render_audio: bool = True,
    render_sr: int = 22050,
    audio_cache: dict[str, np.ndarray] | None = None,
) -> GRPORewardBreakdown:
    """Compute a verifiable [0..1] reward using MIR compatibility + real DSP rendering & QA."""
    cand_by_id = {c.track_id: c for c in task.candidates}
    plan, err = parse_creative_plan(raw_text, set(cand_by_id))
    if plan is None:
        return GRPORewardBreakdown(
            total_reward=0.0,
            schema_valid=False,
            compatibility_reward=0.0,
            dsp_qa_reward=0.0,
            craft_reward=0.0,
            error=err,
        )

    deck_a = task.deck_a
    deck_b = cand_by_id[plan.next_track_id]

    # 1. Harmonic + Tempo + Timbre Compatibility (0.0 .. 0.30)
    cam_dist = camelot_distance(deck_a.camelot, deck_b.camelot)
    harmonic_score = max(0.0, 1.0 - 0.22 * cam_dist)
    stretch_abs = abs((deck_a.bpm - deck_b.bpm) / max(deck_b.bpm, 1.0)) * 100.0
    if stretch_abs <= 4.0:
        tempo_score = 1.0
    elif stretch_abs <= 8.0:
        tempo_score = max(0.0, 1.0 - (stretch_abs - 4.0) / 5.0)
    else:
        tempo_score = 0.0
    timbre_sim = timbre_cosine_similarity(deck_a, deck_b)
    timbre_score = min(1.0, max(0.0, (timbre_sim - 0.3) / 0.65))
    compat_reward = 0.30 * (0.40 * harmonic_score + 0.40 * tempo_score + 0.20 * timbre_score)

    # 2. Phrase & Creative Craft (0.0 .. 0.20)
    # Club blends must be full phrases (>=24 beats); short 8-16b overlaps are only for quick_cut
    if plan.transition_style == "quick_cut":
        phrase_overlap_ok = (
            1.0 if (4 <= plan.overlap_beats <= 16 and plan.overlap_beats % 4 == 0) else 0.0
        )
    else:
        phrase_overlap_ok = (
            1.0
            if (24 <= plan.overlap_beats <= 64 and plan.overlap_beats % 8 == 0)
            else (0.35 if plan.overlap_beats == 16 else 0.0)
        )
    a_max_cue = max(0, len(deck_a.beat_times) - plan.overlap_beats - 4)
    b_max_cue = max(0, len(deck_b.beat_times) - plan.overlap_beats - 4)
    cue_bounds_ok = (
        1.0
        if (16 <= plan.cue_out_beat_prev <= a_max_cue and 0 <= plan.cue_in_beat_next <= b_max_cue)
        else 0.0
    )
    cue_bar_ok = (
        1.0 if (plan.cue_out_beat_prev % 4 == 0 and plan.cue_in_beat_next % 4 == 0) else 0.3
    )
    tech_fit = _creative_technique_fit(deck_a, deck_b, plan, stretch_abs, cam_dist, timbre_sim)
    sec_fit = _section_handoff_score(deck_a, deck_b, plan)
    craft_reward = 0.20 * (
        0.20 * phrase_overlap_ok
        + 0.20 * cue_bounds_ok
        + 0.10 * cue_bar_ok
        + 0.25 * tech_fit
        + 0.25 * sec_fit
    )

    # 3. Real DSP Render + QA Verifier (0.0 .. 0.50)
    qa_flags: tuple[str, ...] = ()
    if render_audio:
        safe_cue_out = min(max(16, plan.cue_out_beat_prev), max(16, a_max_cue))
        safe_cue_in = min(max(0, plan.cue_in_beat_next), max(0, b_max_cue))
        spec = RenderSpec(
            cue_out_beat_a=safe_cue_out,
            cue_in_beat_b=safe_cue_in,
            overlap_beats=plan.overlap_beats,
            transition_style=plan.transition_style,
            bass_swap_beat_frac=plan.bass_swap_beat_frac,
            hat_morph_end_frac=plan.hat_morph_end_frac,
            harmony_strip_a_end_frac=plan.harmony_strip_a_end_frac,
            harmony_entry_b_start_frac=plan.harmony_entry_b_start_frac,
            vocal_entry_b_start_frac=plan.vocal_entry_b_start_frac,
            context_beats=12,
        )
        try:
            y_a = audio_cache.get(deck_a.track_id) if audio_cache is not None else None
            y_b = audio_cache.get(deck_b.track_id) if audio_cache is not None else None
            render_res = render_transition(
                deck_a,
                deck_b,
                spec,
                out_wav=None,
                sr=render_sr,
                stereo=False,
                y_a=y_a,
                y_b=y_b,
            )
            qa = verify_transition(
                render_res, deck_a, deck_b, spec, include_psychoacoustic=True
            )
            dsp_qa_reward = 0.50 * float(qa.verifiable_reward)
            qa_flags = tuple(qa.defects_detected)
        except Exception as exc:
            dsp_qa_reward = 0.0
            qa_flags = (f"render_error:{type(exc).__name__}",)
    else:
        dsp_qa_reward = 0.50 * (0.5 * tempo_score + 0.5 * harmonic_score)

    total = round(min(1.0, max(0.0, compat_reward + craft_reward + dsp_qa_reward)), 4)
    return GRPORewardBreakdown(
        total_reward=total,
        schema_valid=True,
        compatibility_reward=round(compat_reward, 4),
        dsp_qa_reward=round(dsp_qa_reward, 4),
        craft_reward=round(craft_reward, 4),
        plan=plan,
        qa_flags=qa_flags,
    )
