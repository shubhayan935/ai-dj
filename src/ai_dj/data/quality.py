"""Quality gates, transition-style classification, and composite taste/quality ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ai_dj.data.schemas import QualityBreakdown, QualityTier, TransitionStyle


@dataclass(frozen=True)
class QualityConfig:
    """Configurable thresholds for filtering noisy DTW alignments and ranking transitions."""

    min_match_rate: float = 0.50
    max_dtw_cost: float = 0.85
    min_overlap_beats: float = 4.0
    max_overlap_beats: float = 192.0
    min_matched_beats: int = 64
    max_tempo_stretch_pct: float = 10.0
    min_bpm: float = 70.0
    max_bpm: float = 185.0
    min_bpm_stability: float = 0.70
    min_mix_id_ratio: float = 0.50


def classify_transition_style(overlap_beats: float) -> TransitionStyle:
    if overlap_beats < 8.0:
        return "quick_cut"
    if overlap_beats < 24.0:
        return "short_blend"
    if overlap_beats < 64.0:
        return "standard_blend"
    if overlap_beats < 128.0:
        return "long_blend"
    return "extended_overlay"


def compute_tempo_stretch_pct(matched_time_mix: float, matched_time_track: float) -> float:
    """Return percent speed change of track inside the mix (positive = sped up)."""
    if matched_time_track <= 0 or matched_time_mix <= 0:
        return 0.0
    # If the track takes LESS time in the mix than its original duration, it was sped up.
    speed_ratio = matched_time_track / matched_time_mix
    return round((speed_ratio - 1.0) * 100.0, 3)


def _phrase_alignment_score(overlap_beats: float) -> float:
    """Score how cleanly an overlap aligns to musical phrase multiples (8/16/32/64 beats)."""
    if overlap_beats <= 0:
        return 0.0
    canonical_phrases = (8.0, 16.0, 32.0, 48.0, 64.0, 96.0, 128.0)
    nearest_dist = min(abs(overlap_beats - p) for p in canonical_phrases)
    # Within 2 beats of a phrase boundary -> ~1.0; >= 12 beats off -> 0.2 floor
    phrase_prox = max(0.2, 1.0 - (nearest_dist / 12.0))
    # Slight preference for realistic blends (16-96 beats) over extremes
    range_bonus = 1.0 if 12.0 <= overlap_beats <= 128.0 else 0.75
    return round(min(1.0, phrase_prox * range_bonus), 4)


def score_transition(
    *,
    match_rate_prev: float,
    match_rate_next: float,
    cost_prev: float,
    cost_next: float,
    matched_beats_prev: int,
    matched_beats_next: int,
    overlap_beats: float,
    stretch_prev_pct: float,
    stretch_next_pct: float,
    key_change_prev: int,
    key_change_next: int,
    bpm_prev: float | None,
    bpm_next: float | None,
    stability_prev: float | None,
    stability_next: float | None,
    mix_id_ratio: float,
    curated_series: str | None,
    pair_repeat_count: int,
    artist_pair_repeat_count: int,
    config: QualityConfig | None = None,
) -> QualityBreakdown:
    """Evaluate hard technical gates and compute a transparent composite quality score."""
    cfg = config or QualityConfig()
    failures: list[str] = []

    if match_rate_prev < cfg.min_match_rate or match_rate_next < cfg.min_match_rate:
        failures.append("low_match_rate")
    if cost_prev > cfg.max_dtw_cost or cost_next > cfg.max_dtw_cost:
        failures.append("high_dtw_cost")
    if overlap_beats < cfg.min_overlap_beats:
        failures.append("overlap_too_short")
    elif overlap_beats > cfg.max_overlap_beats:
        failures.append("overlap_too_long")
    if matched_beats_prev < cfg.min_matched_beats or matched_beats_next < cfg.min_matched_beats:
        failures.append("too_few_matched_beats")
    if (
        abs(stretch_prev_pct) > cfg.max_tempo_stretch_pct
        or abs(stretch_next_pct) > cfg.max_tempo_stretch_pct
    ):
        failures.append("excessive_tempo_stretch")
    if bpm_prev is None or bpm_next is None:
        failures.append("missing_bpm")
    elif not (cfg.min_bpm <= bpm_prev <= cfg.max_bpm and cfg.min_bpm <= bpm_next <= cfg.max_bpm):
        failures.append("bpm_out_of_range")
    if (
        stability_prev is not None
        and stability_next is not None
        and (stability_prev < cfg.min_bpm_stability or stability_next < cfg.min_bpm_stability)
    ):
        failures.append("unstable_beatgrid")
    if mix_id_ratio < cfg.min_mix_id_ratio:
        failures.append("sparse_tracklist")

    passed = len(failures) == 0

    # 1. Alignment quality (DTW match rate & inverse cost calibrated to cosine-DTW range)
    avg_mr = max(0.0, min(1.0, 0.5 * (match_rate_prev + match_rate_next)))
    avg_cost = max(0.0, 0.5 * (cost_prev + cost_next))
    cost_score = max(0.0, min(1.0, 1.0 - (avg_cost / 1.0)))
    alignment_score = round(0.65 * avg_mr + 0.35 * cost_score, 4)

    # 2. Phrase & overlap quality
    phrase_overlap_score = _phrase_alignment_score(overlap_beats)

    # 3. Tempo & harmonic sanity
    avg_stretch = 0.5 * (abs(stretch_prev_pct) + abs(stretch_next_pct))
    stretch_score = max(0.0, min(1.0, 1.0 - (avg_stretch / 15.0)))
    avg_key_shift = 0.5 * (abs(key_change_prev) + abs(key_change_next))
    key_score = max(0.0, min(1.0, 1.0 - (avg_key_shift / 4.0)))
    tempo_harmonic_score = round(0.65 * stretch_score + 0.35 * key_score, 4)

    # 4. Taste repeatability signal (multiple independent DJs playing the pair / artist bridge)
    pair_sig = min(1.0, math.log1p(max(0, pair_repeat_count - 1)) / math.log1p(5))
    artist_sig = min(1.0, math.log1p(max(0, artist_pair_repeat_count - 1)) / math.log1p(10))
    repeatability_score = round(0.3 + 0.45 * pair_sig + 0.25 * artist_sig, 4)

    # 5. Mix curation & completeness signal
    curation_bonus = 0.30 if curated_series else 0.0
    curation_score = round(min(1.0, 0.70 * max(0.0, min(1.0, mix_id_ratio)) + curation_bonus), 4)

    composite = round(
        0.35 * alignment_score
        + 0.20 * phrase_overlap_score
        + 0.15 * tempo_harmonic_score
        + 0.15 * repeatability_score
        + 0.15 * curation_score,
        4,
    )

    tier: QualityTier = "rejected" if not passed else "bronze"
    return QualityBreakdown(
        passed_hard_gates=passed,
        failure_reasons=failures,
        tier=tier,
        composite_score=composite,
        alignment_score=alignment_score,
        phrase_overlap_score=phrase_overlap_score,
        tempo_harmonic_score=tempo_harmonic_score,
        repeatability_score=repeatability_score,
        curation_score=curation_score,
    )


def assign_quality_tiers(
    records_scores: list[tuple[int, QualityBreakdown]],
) -> dict[int, QualityTier]:
    """Assign gold (top 20%), silver (next 50%), bronze (bottom 30%) among passing transitions."""
    passing = [(idx, q.composite_score) for idx, q in records_scores if q.passed_hard_gates]
    passing.sort(key=lambda x: x[1], reverse=True)
    n = len(passing)
    gold_cut = max(1, int(0.20 * n)) if n > 0 else 0
    silver_cut = max(gold_cut, int(0.70 * n)) if n > 0 else 0

    tiers: dict[int, QualityTier] = {}
    for rank, (idx, _) in enumerate(passing):
        if rank < gold_cut:
            tiers[idx] = "gold"
        elif rank < silver_cut:
            tiers[idx] = "silver"
        else:
            tiers[idx] = "bronze"
    for idx, q in records_scores:
        if not q.passed_hard_gates:
            tiers[idx] = "rejected"
    return tiers
