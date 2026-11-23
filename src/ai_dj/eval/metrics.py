"""Offline retrieval and transition-planning metrics for DJ planners."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from ai_dj.data.schemas import SFTExample, TransitionStyle

ALL_STYLES: tuple[TransitionStyle, ...] = (
    "quick_cut",
    "short_blend",
    "standard_blend",
    "long_blend",
    "extended_overlay",
)


@dataclass(frozen=True)
class PlannerPrediction:
    """Output of any DJ planner (heuristic, zero-shot LLM, or post-trained model)."""

    ranked_track_ids: list[str]
    predicted_overlap_beats: float
    predicted_style: TransitionStyle
    predicted_tempo_stretch_pct: float = 0.0


def _rank_of_target(ranked_ids: list[str], target_id: str) -> int | None:
    for idx, tid in enumerate(ranked_ids, start=1):
        if tid == target_id:
            return idx
    return None


def _macro_f1(y_true: list[str], y_pred: list[str], labels: tuple[str, ...]) -> float:
    tp: Counter[str] = Counter()
    fp: Counter[str] = Counter()
    fn: Counter[str] = Counter()
    for t, p in zip(y_true, y_pred, strict=True):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1

    f1s: list[float] = []
    for lbl in labels:
        denom = 2 * tp[lbl] + fp[lbl] + fn[lbl]
        if denom == 0:
            continue
        f1s.append((2.0 * tp[lbl]) / denom)
    return sum(f1s) / len(f1s) if f1s else 0.0


def _score_subset(
    examples: list[SFTExample],
    predictions: list[PlannerPrediction],
) -> dict[str, float | int]:
    n = len(examples)
    if n == 0:
        return {"count": 0}

    r1 = 0
    r5 = 0
    mrr_sum = 0.0
    ndcg10_sum = 0.0
    overlap_abs_err = 0.0
    overlap_within_8 = 0
    stretch_abs_err = 0.0
    style_true: list[str] = []
    style_pred: list[str] = []

    for ex, pred in zip(examples, predictions, strict=True):
        rank = _rank_of_target(pred.ranked_track_ids, ex.target_track_id)
        if rank is not None:
            if rank == 1:
                r1 += 1
            if rank <= 5:
                r5 += 1
            mrr_sum += 1.0 / rank
            if rank <= 10:
                ndcg10_sum += 1.0 / math.log2(rank + 1.0)

        err_ov = abs(pred.predicted_overlap_beats - ex.target_plan.overlap_beats)
        overlap_abs_err += err_ov
        if err_ov <= 8.0:
            overlap_within_8 += 1

        stretch_abs_err += abs(
            pred.predicted_tempo_stretch_pct - ex.target_plan.tempo_stretch_next_pct
        )
        style_true.append(ex.target_plan.transition_style)
        style_pred.append(pred.predicted_style)

    style_acc = sum(1 for t, p in zip(style_true, style_pred, strict=True) if t == p) / n
    return {
        "count": n,
        "recall_at_1": round(r1 / n, 4),
        "recall_at_5": round(r5 / n, 4),
        "mrr": round(mrr_sum / n, 4),
        "ndcg_at_10": round(ndcg10_sum / n, 4),
        "overlap_beats_mae": round(overlap_abs_err / n, 2),
        "overlap_within_8_beats": round(overlap_within_8 / n, 4),
        "style_accuracy": round(style_acc, 4),
        "style_macro_f1": round(_macro_f1(style_true, style_pred, ALL_STYLES), 4),
        "tempo_stretch_mae_pct": round(stretch_abs_err / n, 2),
    }


def evaluate_predictions(
    examples: list[SFTExample],
    predictions: list[PlannerPrediction],
) -> dict:
    """Compute overall and tier-stratified retrieval & transition-plan metrics."""
    if len(examples) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(examples)} examples vs {len(predictions)} predictions"
        )

    overall = _score_subset(examples, predictions)
    by_tier_ex: dict[str, list[SFTExample]] = defaultdict(list)
    by_tier_pred: dict[str, list[PlannerPrediction]] = defaultdict(list)
    for ex, pred in zip(examples, predictions, strict=True):
        by_tier_ex[ex.tier].append(ex)
        by_tier_pred[ex.tier].append(pred)

    by_tier = {
        tier: _score_subset(by_tier_ex[tier], by_tier_pred[tier])
        for tier in ("gold", "silver", "bronze")
        if tier in by_tier_ex
    }
    return {"overall": overall, "by_tier": by_tier}
