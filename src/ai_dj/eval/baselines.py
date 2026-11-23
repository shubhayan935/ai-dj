"""Deterministic baseline planners for benchmarking post-trained AI DJ models."""

from __future__ import annotations

import random
from typing import Protocol

from ai_dj.data.parsing import normalize_name
from ai_dj.data.quality import classify_transition_style
from ai_dj.data.schemas import CandidateTrackSummary, SFTExample
from ai_dj.eval.metrics import PlannerPrediction


class DJPlanner(Protocol):
    name: str

    def predict(self, example: SFTExample, *, include_dj: bool = True) -> PlannerPrediction: ...


class RandomPlanner:
    """Uniform random candidate ordering + corpus-median transition plan (sanity floor)."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def predict(self, example: SFTExample, *, include_dj: bool = True) -> PlannerPrediction:
        del include_dj
        rng = random.Random(f"{self.seed}:{example.example_id}")
        ids = [c.track_id for c in example.candidates]
        rng.shuffle(ids)
        return PlannerPrediction(
            ranked_track_ids=ids,
            predicted_overlap_beats=48.0,
            predicted_style="standard_blend",
            predicted_tempo_stretch_pct=0.0,
        )


class BPMProximityPlanner:
    """Ranks candidates purely by BPM closeness to the current track."""

    name = "bpm_proximity"

    def predict(self, example: SFTExample, *, include_dj: bool = True) -> PlannerPrediction:
        cur_bpm = example.current_track.bpm or 125.0

        def _dist(cand: CandidateTrackSummary) -> tuple[float, str]:
            if cand.bpm is None:
                return (999.0, cand.track_id)
            return (abs(cand.bpm - cur_bpm), cand.track_id)

        ranked = [c.track_id for c in sorted(example.candidates, key=_dist)]
        top = next((c for c in example.candidates if c.track_id == ranked[0]), None)
        stretch = 0.0
        if top and top.bpm and top.bpm > 0 and example.current_track.bpm:
            stretch = round(100.0 * (example.current_track.bpm - top.bpm) / top.bpm, 2)
        return PlannerPrediction(
            ranked_track_ids=ranked,
            predicted_overlap_beats=48.0,
            predicted_style="standard_blend",
            predicted_tempo_stretch_pct=stretch,
        )


class HeuristicDJPlanner:
    """Transparent rule-based DJ baseline using SetBrief genres, BPM, label affinity, and phase."""

    name = "heuristic_dj"

    def _genre_set(self, genres: list[str]) -> set[str]:
        return {normalize_name(g) for g in genres if g}

    def _predict_overlap_beats(self, example: SFTExample) -> float:
        genres = self._genre_set(example.set_brief.genres or example.current_track.genres)
        base = 48.0
        if genres & {"techno", "minimal", "deep house", "progressive house", "tech house"}:
            base = 64.0
        elif genres & {"drum bass", "dnb", "dubstep", "hardcore", "trap", "hip hop", "big room"}:
            base = 32.0
        if example.set_brief.set_phase in ("opener", "closer"):
            base += 8.0
        return base

    def predict(self, example: SFTExample, *, include_dj: bool = True) -> PlannerPrediction:
        brief = example.set_brief
        if not include_dj:
            brief = brief.model_copy(update={"style_reference_dj": None})

        cur_bpm = example.current_track.bpm or 125.0
        brief_genres = self._genre_set(brief.genres)
        context_genres = self._genre_set(example.current_track.genres)
        for h in example.history:
            context_genres |= self._genre_set(h.genres)

        recent_labels = {
            normalize_name(t.record_label)
            for t in [*example.history, example.current_track]
            if t.record_label
        }
        recent_artists = {
            normalize_name(t.artist)
            for t in [*example.history, example.current_track]
            if t.artist and t.artist != "Unknown"
        }

        def _score(cand: CandidateTrackSummary) -> tuple[float, str]:
            # 1. BPM proximity to current track (0..1)
            if cand.bpm is not None:
                bpm_diff = abs(cand.bpm - cur_bpm)
                bpm_score = max(0.0, 1.0 - (bpm_diff / 12.0))
                if (
                    brief.target_bpm_min is not None
                    and brief.target_bpm_max is not None
                    and brief.target_bpm_min <= cand.bpm <= brief.target_bpm_max
                ):
                    bpm_score += 0.15
            else:
                bpm_score = 0.0

            # 2. Genre overlap with SetBrief and recent history (0..1)
            c_genres = self._genre_set(cand.genres)
            if brief_genres and c_genres:
                brief_jacc = len(brief_genres & c_genres) / len(brief_genres | c_genres)
            else:
                brief_jacc = 0.0
            if context_genres and c_genres:
                ctx_jacc = len(context_genres & c_genres) / len(context_genres | c_genres)
            else:
                ctx_jacc = 0.0
            genre_score = 0.6 * brief_jacc + 0.4 * ctx_jacc

            # 3. Record-label affinity (DJs frequently chain tracks from the same label)
            label_bonus = 0.0
            if cand.record_label and normalize_name(cand.record_label) in recent_labels:
                label_bonus = 0.20

            # 4. Slight penalty for immediate same-artist repeat unless DJ explicitly matches
            artist_norm = normalize_name(cand.artist) if cand.artist else "unknown"
            artist_adj = -0.10 if artist_norm in recent_artists else 0.0
            if (
                include_dj
                and brief.style_reference_dj
                and artist_norm == normalize_name(brief.style_reference_dj)
            ):
                artist_adj = 0.15

            total = 0.45 * bpm_score + 0.45 * genre_score + label_bonus + artist_adj
            return (total, cand.track_id)

        ranked_cands = sorted(example.candidates, key=_score, reverse=True)
        ranked_ids = [c.track_id for c in ranked_cands]

        top = ranked_cands[0] if ranked_cands else None
        stretch = 0.0
        if top and top.bpm and top.bpm > 0 and example.current_track.bpm:
            stretch = round(100.0 * (example.current_track.bpm - top.bpm) / top.bpm, 2)

        overlap = self._predict_overlap_beats(example)
        return PlannerPrediction(
            ranked_track_ids=ranked_ids,
            predicted_overlap_beats=overlap,
            predicted_style=classify_transition_style(overlap),
            predicted_tempo_stretch_pct=stretch,
        )
