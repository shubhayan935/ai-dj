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
    res = render_transition(card_a, card_b, spec, out_wav=wav_path, sr=22050, y_a=y_a, y_b=y_b)
    qa = verify_transition(res, card_a, card_b, spec)

    assert wav_path.exists() and wav_path.stat().st_size > 10_000
    assert qa.passed, f"Expected clean transition to pass QA, got defects: {qa.defects_detected}"
    assert qa.verifiable_reward >= 0.95
    assert qa.beat_aligned
    assert not qa.has_clipping
    assert not qa.has_bass_clash


def test_filter_sweep_and_stem_strip_blend_pass_qa() -> None:
    y_a, card_a = synthesize_electronic_track(bpm=124.0, duration_sec=40.0, seed=11)
    y_b, card_b = synthesize_electronic_track(bpm=125.0, duration_sec=40.0, seed=12)
    for style in ("filter_sweep", "stem_strip_blend", "long_blend"):
        spec = RenderSpec(
            cue_out_beat_a=32,
            cue_in_beat_b=16,
            overlap_beats=16,
            transition_style=style,
            context_beats=12,
        )
        res = render_transition(card_a, card_b, spec, sr=22050, y_a=y_a, y_b=y_b)
        qa = verify_transition(res, card_a, card_b, spec)
        assert qa.passed, f"Style {style} failed QA: {qa.defects_detected}"


def test_defect_gym_meets_95_pct_recall_gate(tmp_path: Path) -> None:
    report = run_defect_gym(out_dir=tmp_path, num_pairs=4)
    assert report["overall_defect_recall"] >= 0.95, report
    assert report["clean_false_positive_rate"] <= 0.10, report
    assert report["passed_exit_gate"] is True
    assert (tmp_path / "defect_gym_report.md").exists()


def test_psychoacoustic_critic_detects_harmonic_clash_and_dual_drums() -> None:
    import numpy as np

    from ai_dj.audio.critic import evaluate_transition_psychoacoustics, plomp_levelt_roughness

    sr = 22050
    t = np.linspace(0.0, 4.0, sr * 4, endpoint=False, dtype=np.float32)
    # 1. Minor 2nd (440 Hz + 466.16 Hz) has much higher Plomp-Levelt roughness than unison/fifth
    consonant = 0.4 * np.sin(2 * np.pi * 440.0 * t[:sr]) + 0.4 * np.sin(2 * np.pi * 659.25 * t[:sr])
    dissonant = 0.4 * np.sin(2 * np.pi * 440.0 * t[:sr]) + 0.4 * np.sin(2 * np.pi * 466.16 * t[:sr])
    assert plomp_levelt_roughness(dissonant, sr) > plomp_levelt_roughness(consonant, sr) * 2.0

    # 2. Clashing melodies (A4 + Bb4 simultaneously) + dual simultaneous drums
    harm_a = 0.35 * np.sin(2 * np.pi * 440.0 * t)
    harm_b = 0.35 * np.sin(2 * np.pi * 466.16 * t)
    rng = np.random.default_rng(7)
    drums_a = (0.3 * rng.standard_normal(len(t))).astype(np.float32)
    drums_b = (0.3 * rng.standard_normal(len(t))).astype(np.float32)
    mix_bad = harm_a + harm_b + drums_a + drums_b
    crit_bad = evaluate_transition_psychoacoustics(
        mix_bad,
        harm_a + drums_a,
        harm_b + drums_b,
        sr,
        rendered_a_drums=drums_a,
        rendered_b_drums=drums_b,
        rendered_a_harm=harm_a,
        rendered_b_harm=harm_b,
    )
    assert "harmonic_clash" in crit_bad.issues
    assert "dual_drum_overlap" in crit_bad.issues
    assert len(crit_bad.actionable_fixes) >= 2

    # 3. Clean drum-bridge separation (A fades out before B enters -> zero clash)
    half = len(t) // 2
    ha_clean = np.concatenate([harm_a[:half], np.zeros(len(t) - half, dtype=np.float32)])
    hb_clean = np.concatenate([np.zeros(half, dtype=np.float32), harm_b[half:]])
    da_clean = np.concatenate([drums_a[:half], np.zeros(len(t) - half, dtype=np.float32)])
    db_clean = np.concatenate([np.zeros(half, dtype=np.float32), drums_b[half:]])
    crit_good = evaluate_transition_psychoacoustics(
        ha_clean + hb_clean + da_clean + db_clean,
        ha_clean + da_clean,
        hb_clean + db_clean,
        sr,
        rendered_a_drums=da_clean,
        rendered_b_drums=db_clean,
        rendered_a_harm=ha_clean,
        rendered_b_harm=hb_clean,
    )
    assert crit_good.peak_harmonic_clash < 0.05
    assert crit_good.peak_dual_drum_overlap < 0.05
    assert crit_good.naturalness_score > crit_bad.naturalness_score


def test_cdj_loop_and_echo_out_primitives() -> None:
    """Beat-exact Deck A looping & echo-out produce periodic loops and decaying tails."""
    import numpy as np

    from ai_dj.audio.renderer import _apply_beat_synced_echo_out, _build_beat_exact_loop
    from ai_dj.rl.grpo_env import parse_creative_plan

    sr = 22050
    beat_dur = 0.5  # 120 BPM
    loop_n = int(round(4 * beat_dur * sr))  # 4-beat loop
    total_n = loop_n * 4
    t = np.arange(total_n, dtype=np.float32) / sr
    src = np.vstack([np.sin(2.0 * np.pi * 220.0 * t), np.sin(2.0 * np.pi * 330.0 * t)])
    # Mark a distinct impulse in beat 1 of the loop
    src[:, 500:700] += 0.5

    looped = _build_beat_exact_loop(src, loop_n, total_n, sr, tighten=False)
    assert looped.shape == (2, total_n)
    # Second repetition should match first repetition inside the body
    np.testing.assert_allclose(
        looped[:, 400:1200], looped[:, loop_n + 400 : loop_n + 1200], atol=1e-4
    )

    tight = _build_beat_exact_loop(src, loop_n, total_n, sr, tighten=True)
    assert tight.shape == (2, total_n)
    assert np.isfinite(tight).all()

    echoed = _apply_beat_synced_echo_out(src, sr, beat_dur, cut_frac=0.5, feedback=0.55)
    assert echoed.shape == (2, total_n)
    assert np.isfinite(echoed).all()
    cut_idx = total_n // 2
    # Tail after cut_frac should have decaying echo energy (not silent, and lower than pre-cut)
    pre_rms = float(np.sqrt(np.mean(echoed[:, :cut_idx] ** 2)))
    post_rms = float(np.sqrt(np.mean(echoed[:, cut_idx : cut_idx + sr] ** 2)))
    assert 0.05 * pre_rms < post_rms < 0.95 * pre_rms

    # Schema parsing supports new CDJ primitives + defaults when omitted
    raw = (
        '{"next_track_id": "t2", "transition_style": "stem_strip_blend", '
        '"overlap_beats": 32, "cue_out_beat_prev": 128, "cue_in_beat_next": 32, '
        '"bass_swap_beat_frac": 0.5, "loop_a_beats": 8, "loop_a_tighten": true, '
        '"hold_a_stem": "loop_all", "exit_fx": "hpf_echo_out"}'
    )
    plan, err = parse_creative_plan(raw, {"t2"})
    assert err is None and plan is not None
    assert plan.loop_a_beats == 8
    assert plan.loop_a_tighten is True
    assert plan.hold_a_stem == "loop_all"
    assert plan.exit_fx == "hpf_echo_out"


def test_composable_cdj_dsl_execution() -> None:
    """Composable CDJ DSL executes simple drop-swap and complex B-tease+reverb+echo programs."""
    import numpy as np

    from ai_dj.audio.cdj_dsl import (
        DeckLoop,
        apply_deck_loop,
        build_b_tease_acapella_bridge_dsl,
        build_drop_swap_dsl,
        execute_cdj_program,
    )

    sr = 22050
    beat_dur = 0.5
    ov_n = int(round(32 * beat_dur * sr))
    t = np.arange(ov_n, dtype=np.float32) / sr

    def _fake_stems(f_base: float) -> dict[str, np.ndarray]:
        sig = np.vstack([np.sin(2.0 * np.pi * f_base * t), np.sin(2.0 * np.pi * (f_base + 5) * t)])
        return {
            "drums": (0.4 * sig).astype(np.float32),
            "bass": (0.3 * sig).astype(np.float32),
            "vocals": (0.25 * sig).astype(np.float32),
            "other": (0.25 * sig).astype(np.float32),
        }

    stems_a = _fake_stems(220.0)
    stems_b = _fake_stems(330.0)

    # 1. Complex Deck B tease loop + acapella bridge + reverb wash + echo-out
    prog_complex = build_b_tease_acapella_bridge_dsl(overlap_beats=32)
    assert prog_complex.deck_b.loop is not None
    looped_b = apply_deck_loop(
        stems_b["drums"], DeckLoop(length_beats=4, end_beat=16.0), sr, beat_dur
    )
    assert looped_b.shape == stems_b["drums"].shape and np.isfinite(looped_b).all()

    ra, rb, da, db, ha, hb = execute_cdj_program(prog_complex, stems_a, stems_b, sr, beat_dur)
    for arr in (ra, rb, da, db, ha, hb):
        assert arr.shape == (2, ov_n)
        assert np.isfinite(arr).all()
    # Deck A should be dominant early, Deck B dominant late
    assert float(np.sqrt(np.mean(ra[:, : sr * 2] ** 2))) > float(
        np.sqrt(np.mean(ra[:, -sr * 2 :] ** 2))
    )
    assert float(np.sqrt(np.mean(rb[:, -sr * 2 :] ** 2))) > float(
        np.sqrt(np.mean(rb[:, : sr * 2] ** 2))
    )

    # 2. Simple phrase drop-swap
    prog_simple = build_drop_swap_dsl(overlap_beats=16)
    ov_16 = int(round(16 * beat_dur * sr))
    sa16 = {k: v[:, :ov_16] for k, v in stems_a.items()}
    sb16 = {k: v[:, :ov_16] for k, v in stems_b.items()}
    ra2, rb2, *_ = execute_cdj_program(prog_simple, sa16, sb16, sr, beat_dur)
    assert ra2.shape == (2, ov_16) and rb2.shape == (2, ov_16)
    assert np.isfinite(ra2).all() and np.isfinite(rb2).all()
