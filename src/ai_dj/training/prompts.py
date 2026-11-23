"""Canonical prompt formatting and structured JSON completion parsing for AI DJ post-training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ai_dj.data.schemas import (
    CandidateTrackSummary,
    DPOPair,
    SetBrief,
    SFTExample,
    TransitionPlan,
    TransitionStyle,
)
from ai_dj.eval.metrics import ALL_STYLES, PlannerPrediction

SYSTEM_PROMPT = (
    "You are an expert club DJ planner. Given a SetBrief (genres, target BPM range, set phase, "
    "optional reference DJ), recent track history, the currently playing track, and a candidate "
    "crate, choose the single best next track and output an exact symbolic transition plan.\n"
    "Prioritize musical/genre compatibility, coherent set pacing, tight tempo matching, "
    "phrase-aligned cues (multiples of 4/8/16 beats), and an appropriate blend style and length.\n"
    "Respond with ONLY a single valid JSON object with keys in order: "
    '"next_track_id", "transition_style", "overlap_beats", '
    '"cue_out_beat_prev", "cue_in_beat_next", "tempo_stretch_next_pct". '
    'Valid transition_style values: "quick_cut", "short_blend", "standard_blend", '
    '"long_blend", "extended_overlay".'
)

STYLE_PHRASE_BUCKETS: dict[TransitionStyle, tuple[int, ...]] = {
    "quick_cut": (4,),
    "short_blend": (8, 16),
    "standard_blend": (24, 32, 48),
    "long_blend": (64, 96),
    "extended_overlay": (128, 160),
}


def quantize_overlap_for_style(overlap_beats: float, style: TransitionStyle) -> int:
    """Snap overlap length to the nearest DJ phrase bucket valid for the given style."""
    buckets = STYLE_PHRASE_BUCKETS[style]
    return min(buckets, key=lambda b: abs(b - overlap_beats))


def _quantize_bar_beat(beat: float) -> int:
    """Snap a beat position to the nearest 4-beat bar boundary."""
    return max(0, int(round(max(0.0, float(beat)) / 4.0) * 4))


class ModelMixPlanOutput(BaseModel):
    """Canonical structured JSON output emitted by the post-trained DJ planner."""

    next_track_id: str
    transition_style: TransitionStyle
    overlap_beats: float = Field(ge=0.0)
    cue_out_beat_prev: float = Field(ge=0.0)
    cue_in_beat_next: float = Field(ge=0.0)
    tempo_stretch_next_pct: float = 0.0


@dataclass(frozen=True)
class ParseDiagnostics:
    """Validity metadata captured when parsing raw LLM output."""

    json_valid: bool
    valid_track_id: bool
    style_valid: bool
    error: str | None = None


def _fmt_track_line(track: CandidateTrackSummary, prefix: str = "") -> str:
    parts: list[str] = []
    if prefix:
        parts.append(prefix)
    parts.append(f"id={track.track_id}")
    title_str = f"{track.artist} - {track.title}"
    if track.version:
        title_str += f" ({track.version})"
    parts.append(title_str)
    if track.record_label:
        parts.append(f"[{track.record_label}]")
    if track.bpm is not None:
        parts.append(f"BPM={track.bpm:.1f}")
    if track.duration_sec is not None:
        mins, secs = divmod(int(round(track.duration_sec)), 60)
        parts.append(f"dur={mins}:{secs:02d}")
    if track.genres:
        parts.append("genres=" + ",".join(track.genres[:4]))
    return " | ".join(parts)


def format_user_prompt(
    set_brief: SetBrief,
    history: list[CandidateTrackSummary],
    current_track: CandidateTrackSummary,
    candidates: list[CandidateTrackSummary],
    *,
    include_dj: bool = True,
) -> str:
    """Render a compact, deterministic user prompt shared across SFT, DPO, and evaluation."""
    brief = (
        set_brief
        if include_dj
        else set_brief.model_copy(update={"style_reference_dj": None})
    )

    lines: list[str] = ["### SetBrief"]
    if brief.genres:
        lines.append(f"- Genres: {', '.join(brief.genres)}")
    if brief.curated_series:
        lines.append(f"- Series: {brief.curated_series}")
    if brief.target_bpm_min is not None and brief.target_bpm_max is not None:
        lines.append(f"- Target BPM: {brief.target_bpm_min:.1f}–{brief.target_bpm_max:.1f}")
    lines.append(f"- Set Phase: {brief.set_phase} ({brief.set_progress * 100:.0f}% through set)")
    if brief.style_reference_dj:
        lines.append(f"- Style Reference DJ: {brief.style_reference_dj}")

    if history:
        lines.append("")
        lines.append("### Recent History (oldest -> newest)")
        for i, tr in enumerate(history, start=1):
            lines.append(f"  H{i}. {_fmt_track_line(tr)}")

    lines.append("")
    lines.append("### Current Track (Deck A)")
    lines.append(f"  {_fmt_track_line(current_track)}")

    lines.append("")
    lines.append(f"### Candidate Crate ({len(candidates)} tracks)")
    for idx, cand in enumerate(candidates, start=1):
        lines.append(f"  {idx:02d}. {_fmt_track_line(cand)}")

    lines.append("")
    lines.append("Select the next track and return the transition plan JSON.")
    return "\n".join(lines)


def format_target_completion(
    track_id: str,
    plan: TransitionPlan,
    *,
    quantize_phrase: bool = True,
) -> str:
    """Serialize ground-truth track choice + symbolic transition plan as compact JSON.

    Places `transition_style` before `overlap_beats` so greedy decoding selects the
    macro blend style first, and optionally snaps overlaps/cues to DJ phrase/bar grids.
    """
    if quantize_phrase:
        overlap_val: float = float(
            quantize_overlap_for_style(plan.overlap_beats, plan.transition_style)
        )
        cue_out_val: float = float(_quantize_bar_beat(plan.track_cue_out_beat_prev))
        cue_in_val: float = float(_quantize_bar_beat(plan.track_cue_in_beat_next))
    else:
        overlap_val = round(max(0.0, float(plan.overlap_beats)), 1)
        cue_out_val = round(max(0.0, float(plan.track_cue_out_beat_prev)), 1)
        cue_in_val = round(max(0.0, float(plan.track_cue_in_beat_next)), 1)

    stretch_val = round(float(plan.tempo_stretch_next_pct), 1)
    if stretch_val == 0.0:
        stretch_val = 0.0

    payload = ModelMixPlanOutput(
        next_track_id=track_id,
        transition_style=plan.transition_style,
        overlap_beats=overlap_val,
        cue_out_beat_prev=cue_out_val,
        cue_in_beat_next=cue_in_val,
        tempo_stretch_next_pct=stretch_val,
    )
    return json.dumps(payload.model_dump(), separators=(",", ":"))


def format_sft_messages(
    example: SFTExample,
    *,
    include_dj: bool = True,
) -> list[dict[str, str]]:
    """Build a 3-turn chat conversation [system, user, assistant] for SFT."""
    user_content = format_user_prompt(
        set_brief=example.set_brief,
        history=example.history,
        current_track=example.current_track,
        candidates=example.candidates,
        include_dj=include_dj,
    )
    assistant_content = format_target_completion(example.target_track_id, example.target_plan)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def format_dpo_pair_messages(
    pair: DPOPair,
    *,
    include_dj: bool = True,
) -> dict[str, object]:
    """Build a DPO preference record with shared prompt and chosen/rejected completions."""
    user_content = format_user_prompt(
        set_brief=pair.set_brief,
        history=pair.history,
        current_track=pair.current_track,
        candidates=pair.candidates,
        include_dj=include_dj,
    )
    return {
        "pair_id": pair.pair_id,
        "split": pair.split,
        "negative_type": pair.negative_type,
        "mix_id": pair.mix_id,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "chosen": [
            {
                "role": "assistant",
                "content": format_target_completion(pair.chosen_track_id, pair.chosen_plan),
            }
        ],
        "rejected": [
            {
                "role": "assistant",
                "content": format_target_completion(
                    pair.rejected_track_id,
                    pair.rejected_plan,
                    quantize_phrase=(pair.negative_type != "off_phrase_plan"),
                ),
            }
        ],
    }


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", flags=re.DOTALL | re.IGNORECASE)


def _extract_json_object(raw_text: str) -> dict | None:
    cleaned = _THINK_RE.sub("", raw_text).strip()
    fence_match = _FENCE_RE.search(cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
    return None


def parse_model_output(
    raw_text: str,
    candidates: list[CandidateTrackSummary],
) -> tuple[PlannerPrediction, ParseDiagnostics]:
    """Parse raw model text into a PlannerPrediction plus schema-validity diagnostics."""
    valid_ids = {c.track_id for c in candidates}

    obj = _extract_json_object(raw_text)
    if obj is None:
        return (
            PlannerPrediction(
                ranked_track_ids=[],
                predicted_overlap_beats=None,
                predicted_style=None,
                predicted_tempo_stretch_pct=None,
            ),
            ParseDiagnostics(
                json_valid=False,
                valid_track_id=False,
                style_valid=False,
                error="Could not parse JSON object from completion",
            ),
        )

    ranked: list[str] = []
    seen: set[str] = set()

    primary_id = str(obj.get("next_track_id", "")).strip()
    valid_primary = primary_id in valid_ids
    if valid_primary:
        ranked.append(primary_id)
        seen.add(primary_id)

    extra_ranked = obj.get("ranked_track_ids") or obj.get("top_5_track_ids")
    if isinstance(extra_ranked, list):
        for item in extra_ranked:
            tid = str(item).strip()
            if tid in valid_ids and tid not in seen:
                ranked.append(tid)
                seen.add(tid)

    overlap: float | None = None
    if "overlap_beats" in obj and obj["overlap_beats"] is not None:
        try:
            ov = float(obj["overlap_beats"])
            if 0.0 <= ov <= 512.0:
                overlap = ov
        except (TypeError, ValueError):
            overlap = None

    raw_style = str(obj.get("transition_style", "")).strip()
    style_valid = raw_style in ALL_STYLES
    style: TransitionStyle | None = (
        raw_style if style_valid else None  # type: ignore[assignment]
    )

    stretch: float | None = None
    if "tempo_stretch_next_pct" in obj and obj["tempo_stretch_next_pct"] is not None:
        try:
            st = float(obj["tempo_stretch_next_pct"])
            if -50.0 <= st <= 50.0:
                stretch = st
        except (TypeError, ValueError):
            stretch = None

    cue_out: float | None = None
    if "cue_out_beat_prev" in obj and obj["cue_out_beat_prev"] is not None:
        try:
            co = float(obj["cue_out_beat_prev"])
            if 0.0 <= co <= 5000.0:
                cue_out = co
        except (TypeError, ValueError):
            cue_out = None

    cue_in: float | None = None
    if "cue_in_beat_next" in obj and obj["cue_in_beat_next"] is not None:
        try:
            ci = float(obj["cue_in_beat_next"])
            if 0.0 <= ci <= 5000.0:
                cue_in = ci
        except (TypeError, ValueError):
            cue_in = None

    return (
        PlannerPrediction(
            ranked_track_ids=ranked,
            predicted_overlap_beats=overlap,
            predicted_style=style,
            predicted_tempo_stretch_pct=stretch,
            predicted_cue_out_beat_prev=cue_out,
            predicted_cue_in_beat_next=cue_in,
        ),
        ParseDiagnostics(
            json_valid=True,
            valid_track_id=valid_primary,
            style_valid=style_valid,
        ),
    )
