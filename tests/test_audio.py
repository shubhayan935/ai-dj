from pathlib import Path

from ai_dj.audio.analyzer import camelot_distance
from ai_dj.audio.defect_gym import run_defect_gym, synthesize_electronic_track
from ai_dj.audio.renderer import RenderSpec, render_transition
from ai_dj.audio.verifier import verify_transition


def test_camelot_distance() -> None:
    assert camelot_distance("8A", "8A") == 0
    assert camelot_distance("8A", "9A") == 1
    assert camelot_distance("8A", "8B") == 1
    assert camelot_distance("12A", "1A") == 1
    assert camelot_distance("8A", "2A") == 6


def test_clean_render_passes_qa_and_writes_wav(tmp_path: Path) -> None:
    y_a, card_a = synthesize_electronic_track(bpm=124.0, duration_sec=40.0, seed=1)
    y_b, card_b = synthesize_electronic_track(bpm=126.0, duration_sec=40.0, seed=2)
    spec = RenderSpec(
        cue_out_beat_a=32,
        cue_in_beat_b=16,
        overlap_beats=16,
        transition_style="bass_swap",
        context_beats=12,
    )
    wav_path = tmp_path / "clean_mix.wav"
    res = render_transition(card_a, card_b, spec, out_wav=wav_path, y_a=y_a, y_b=y_b)
    qa = verify_transition(res, card_a, card_b, spec)

    assert wav_path.exists() and wav_path.stat().st_size > 10_000
    assert qa.passed, f"Expected clean transition to pass QA, got defects: {qa.defects_detected}"
    assert qa.verifiable_reward >= 0.95
    assert qa.beat_aligned
    assert not qa.has_clipping
    assert not qa.has_bass_clash


def test_defect_gym_meets_95_pct_recall_gate(tmp_path: Path) -> None:
    report = run_defect_gym(out_dir=tmp_path, num_pairs=4)
    assert report["overall_defect_recall"] >= 0.95, report
    assert report["clean_false_positive_rate"] <= 0.10, report
    assert report["passed_exit_gate"] is True
    assert (tmp_path / "defect_gym_report.md").exists()
