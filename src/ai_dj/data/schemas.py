"""Canonical data schemas for the AI DJ data, training, and evaluation pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TransitionStyle = Literal[
    "quick_cut",        # < 8 beats
    "short_blend",      # 8 - 24 beats
    "standard_blend",   # 24 - 64 beats
    "long_blend",       # 64 - 128 beats
    "extended_overlay", # > 128 beats
]

QualityTier = Literal["gold", "silver", "bronze", "rejected"]
SplitName = Literal["train", "val", "test"]
SetPhase = Literal["opener", "build", "peak", "cooldown", "closer"]


class SetBrief(BaseModel):
    """Taste and pacing profile conditioning the AI DJ planner."""

    genres: list[str] = Field(default_factory=list)
    curated_series: str | None = None
    target_bpm_min: float | None = None
    target_bpm_max: float | None = None
    set_progress: float = Field(default=0.5, ge=0.0, le=1.0)
    set_phase: SetPhase = "peak"
    style_reference_dj: str | None = None


class TrackCard(BaseModel):
    """Structured metadata and MIR summary for a single track."""

    track_id: str
    raw_title: str = ""
    artist: str = "Unknown"
    title: str = "Unknown"
    version: str | None = None
    record_label: str | None = None
    bpm: float | None = None
    bpm_stability: float | None = Field(default=None, ge=0.0, le=1.0)
    duration_sec: float | None = None
    beat_count: int | None = None
    genres: list[str] = Field(default_factory=list)
    play_count: int = 0
    unique_mix_count: int = 0
    youtube_url: str | None = None


class MixSet(BaseModel):
    """Metadata for a recorded human DJ set."""

    mix_id: str
    title: str
    url: str | None = None
    audio_source: str | None = None
    audio_url: str | None = None
    dj_name: str = "Unknown"
    dj_name_norm: str = "unknown"
    venue_or_show: str | None = None
    mix_date: str | None = None
    year: int | None = None
    tags: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    num_total_tracks: int = 0
    num_identified_tracks: int = 0
    num_available_transitions: int = 0
    identification_ratio: float = 0.0
    curated_series: str | None = None


class TransitionPlan(BaseModel):
    """Symbolic transition plan executed by a deterministic DSP mixer."""

    track_cue_out_beat_prev: float
    track_cue_in_beat_next: float
    overlap_beats: float
    overlap_bars: float
    tempo_stretch_prev_pct: float
    tempo_stretch_next_pct: float
    key_change_prev_semitones: int = 0
    key_change_next_semitones: int = 0
    transition_style: TransitionStyle


class QualityBreakdown(BaseModel):
    """Transparent per-transition quality gates and composite ranking scores."""

    passed_hard_gates: bool
    failure_reasons: list[str] = Field(default_factory=list)
    tier: QualityTier
    composite_score: float = Field(ge=0.0, le=1.0)
    alignment_score: float = Field(ge=0.0, le=1.0)
    phrase_overlap_score: float = Field(ge=0.0, le=1.0)
    tempo_harmonic_score: float = Field(ge=0.0, le=1.0)
    repeatability_score: float = Field(ge=0.0, le=1.0)
    curation_score: float = Field(ge=0.0, le=1.0)


class TransitionRecord(BaseModel):
    """Cleaned, scored transition joining mix context, track cards, and plan."""

    tran_id: str
    mix_id: str
    i_tran: int
    dj_name: str
    dj_name_norm: str
    mix_year: int | None = None
    mix_tags: list[str] = Field(default_factory=list)
    curated_series: str | None = None
    prev_track: TrackCard
    next_track: TrackCard
    match_rate_prev: float
    match_rate_next: float
    cost_prev: float
    cost_next: float
    matched_beats_prev: int
    matched_beats_next: int
    bpm_delta: float | None = None
    pair_repeat_count: int = 1
    artist_pair_repeat_count: int = 1
    plan: TransitionPlan
    quality: QualityBreakdown
    split: SplitName | None = None


class CandidateTrackSummary(BaseModel):
    """Compact candidate representation shown to the planner model."""

    track_id: str
    artist: str
    title: str
    version: str | None = None
    record_label: str | None = None
    bpm: float | None = None
    duration_sec: float | None = None
    genres: list[str] = Field(default_factory=list)


class SFTExample(BaseModel):
    """Training/eval example for next-track selection + symbolic mix planning."""

    example_id: str
    split: SplitName
    tier: QualityTier
    composite_score: float
    mix_id: str
    dj_name: str
    set_brief: SetBrief
    history: list[CandidateTrackSummary]
    current_track: CandidateTrackSummary
    candidates: list[CandidateTrackSummary]
    target_track_id: str
    target_plan: TransitionPlan


class DPOPair(BaseModel):
    """Preference pair for taste and transition-quality alignment."""

    pair_id: str
    split: SplitName
    negative_type: Literal["bpm_matched_distractor", "off_phrase_plan"]
    mix_id: str
    set_brief: SetBrief
    history: list[CandidateTrackSummary]
    current_track: CandidateTrackSummary
    candidates: list[CandidateTrackSummary]
    chosen_track_id: str
    chosen_plan: TransitionPlan
    rejected_track_id: str
    rejected_plan: TransitionPlan
