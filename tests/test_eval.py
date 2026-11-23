"""Tests for offline retrieval and transition-planning eval metrics and baselines."""

from ai_dj.data.schemas import CandidateTrackSummary, SetBrief, SFTExample, TransitionPlan
from ai_dj.eval.baselines import BPMProximityPlanner, HeuristicDJPlanner, RandomPlanner
from ai_dj.eval.metrics import PlannerPrediction, evaluate_predictions


def _make_example(
    *,
    example_id: str = "mix1-01",
    tier: str = "gold",
    target_id: str = "t_target",
    target_bpm: float = 128.0,
    target_genres: list[str] | None = None,
    target_label: str | None = "Drumcode",
    overlap_beats: float = 64.0,
    style: str = "long_blend",
) -> SFTExample:
    cur = CandidateTrackSummary(
        track_id="t_prev",
        artist="Adam Beyer",
        title="Your Mind",
        record_label="Drumcode",
        bpm=128.0,
        genres=["Techno"],
    )
    target = CandidateTrackSummary(
        track_id=target_id,
        artist="Bart Skils",
        title="Dark Mode",
        record_label=target_label,
        bpm=target_bpm,
        genres=target_genres or ["Techno"],
    )
    distractor_a = CandidateTrackSummary(
        track_id="t_dist_1",
        artist="Other Artist",
        title="House Track",
        record_label="Defected",
        bpm=122.0,
        genres=["Deep House"],
    )
    distractor_b = CandidateTrackSummary(
        track_id="t_dist_2",
        artist="Third Artist",
        title="Trance Track",
        record_label="Armada",
        bpm=138.0,
        genres=["Trance"],
    )
    return SFTExample(
        example_id=example_id,
        split="val",
        tier=tier,
        composite_score=0.82,
        mix_id="mix1",
        dj_name="Adam Beyer",
        set_brief=SetBrief(
            genres=["Techno"],
            target_bpm_min=126.0,
            target_bpm_max=132.0,
            set_progress=0.5,
            set_phase="peak",
            style_reference_dj="Adam Beyer",
        ),
        history=[],
        current_track=cur,
        candidates=[distractor_a, target, distractor_b],
        target_track_id=target_id,
        target_plan=TransitionPlan(
            track_cue_out_beat_prev=320.0,
            track_cue_in_beat_next=0.0,
            overlap_beats=overlap_beats,
            overlap_bars=overlap_beats / 4.0,
            tempo_stretch_prev_pct=0.0,
            tempo_stretch_next_pct=0.0,
            transition_style=style,
        ),
    )


def test_evaluate_predictions_perfect_and_imperfect() -> None:
    ex1 = _make_example(example_id="e1", tier="gold", overlap_beats=64.0, style="long_blend")
    ex2 = _make_example(example_id="e2", tier="silver", overlap_beats=32.0, style="standard_blend")

    pred1 = PlannerPrediction(
        ranked_track_ids=["t_target", "t_dist_1", "t_dist_2"],
        predicted_overlap_beats=64.0,
        predicted_style="long_blend",
        predicted_tempo_stretch_pct=0.0,
    )
    pred2 = PlannerPrediction(
        ranked_track_ids=["t_dist_1", "t_target", "t_dist_2"],
        predicted_overlap_beats=48.0,
        predicted_style="standard_blend",
        predicted_tempo_stretch_pct=1.0,
    )

    res = evaluate_predictions([ex1, ex2], [pred1, pred2])
    overall = res["overall"]
    assert overall["count"] == 2
    assert overall["recall_at_1"] == 0.5
    assert overall["recall_at_5"] == 1.0
    assert overall["mrr"] == 0.75
    assert overall["overlap_beats_mae"] == 8.0
    assert overall["style_accuracy"] == 1.0
    assert "gold" in res["by_tier"] and "silver" in res["by_tier"]


def test_heuristic_and_bpm_baselines_rank_matching_track_first() -> None:
    ex = _make_example()
    bpm_pred = BPMProximityPlanner().predict(ex, include_dj=False)
    heur_pred_with = HeuristicDJPlanner().predict(ex, include_dj=True)
    heur_pred_without = HeuristicDJPlanner().predict(ex, include_dj=False)
    rand_pred = RandomPlanner(seed=1).predict(ex, include_dj=True)
    rand_pred_without = RandomPlanner(seed=1).predict(ex, include_dj=False)

    assert bpm_pred.ranked_track_ids[0] == "t_target"
    assert heur_pred_with.ranked_track_ids[0] == "t_target"
    assert heur_pred_without.ranked_track_ids[0] == "t_target"
    assert heur_pred_without.predicted_overlap_beats == 64.0
    assert set(rand_pred.ranked_track_ids) == {"t_target", "t_dist_1", "t_dist_2"}
    assert rand_pred.ranked_track_ids == rand_pred_without.ranked_track_ids
