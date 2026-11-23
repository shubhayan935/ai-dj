"""Canonical prompt formatting and structured JSON completion parsing for AI DJ post-training."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ai_dj.data.quality import classify_transition_style
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
    "phrase-aligned cues (multiples of 8/16/32 beats), and an appropriate blend length and style.\n"
    "Respond with ONLY a single valid JSON object with keys: "
    '"next_track_id", "cue_out_beat_prev", "cue_in_beat_next", "overlap_beats", '
    '"transition_style", "tempo_stretch_next_pct". '
    'Valid transition_style values: "quick_cut", "short_blend", "standard_blend", '
    '"long_blend", "extended_overlay".'
)


class ModelMixPlanOutput(BaseModel):
    """Canonical structured JSON output emitted by the post-trained DJ planner."""

    next_track_id: str
    cue_out_beat_prev: float = Field(ge=0.0)
    cue_in_beat_next: float = Field(ge=0.0)
    overlap_beats: float = Field(ge=0.0)
    transition_style: TransitionStyle
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


def format_target_completion(track_id: str, plan: TransitionPlan) -> str:
    """Serialize ground-truth track choice + symbolic transition plan as compact JSON."""
    payload = ModelMixPlanOutput(
        next_track_id=track_id,
        cue_out_beat_prev=round(max(0.0, float(plan.track_cue_out_beat_prev)), 1),
        cue_in_beat_next=round(max(0.0, float(plan.track_cue_in_beat_next)), 1),
        overlap_beats=round(max(0.0, float(plan.overlap_beats)), 1),
        transition_style=plan.transition_style,
        tempo_stretch_next_pct=round(float(plan.tempo_stretch_next_pct), 2),
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
                "content": format_target_completion(pair.rejected_track_id, pair.rejected_plan),
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
    *,
    default_overlap_beats: float = 48.0,
) -> tuple[PlannerPrediction, ParseDiagnostics]:
    """Parse raw model text into a PlannerPrediction plus schema-validity diagnostics."""
    valid_ids = {c.track_id for c in candidates}
    cand_order = [c.track_id for c in candidates]

    obj = _extract_json_object(raw_text)
    if obj is None:
        return (
            PlannerPrediction(
                ranked_track_ids=[],
                predicted_overlap_beats=default_overlap_beats,
                predicted_style="standard_blend",
                predicted_tempo_stretch_pct=0.0,
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

    # If the model only emitted a single next_track_id, keep remaining candidates out of top-1
    # without artificially inflating Recall@5 beyond what was explicitly predicted.
    if not ranked and cand_order:
        ranked = []

    try:
        overlap = float(obj.get("overlap_beats", default_overlap_beats))
        if not (0.0 <= overlap <= 512.0):
            overlap = default_overlap_beats
    except (TypeError, ValueError):
        overlap = default_overlap_beats

    raw_style = str(obj.get("transition_style", "")).strip()
    style_valid = raw_style in ALL_STYLES
    style: TransitionStyle = (
        raw_style if style_valid else classify_transition_style(overlap)  # type: ignore[assignment]
    )

    try:
        stretch = float(obj.get("tempo_stretch_next_pct", 0.0))
        if not (-50.0 <= stretch <= 50.0):
            stretch = 0.0
    except (TypeError, ValueError):
        stretch = 0.0

    return (
        PlannerPrediction(
            ranked_track_ids=ranked,
            predicted_overlap_beats=overlap,
            predicted_style=style,
            predicted_tempo_stretch_pct=stretch,
        ),
        ParseDiagnostics(
            json_valid=True,
            valid_track_id=valid_primary,
            style_valid=style_valid,
        ),
    )
