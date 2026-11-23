"""Defect Gym: injects known audio defects into clean DJ transitions to calibrate QA recall."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ai_dj.audio.analyzer import AudioTrackCard, analyze_track
from ai_dj.audio.renderer import RenderResult, RenderSpec, render_transition
from ai_dj.audio.verifier import QAThresholds, verify_transition

DEFECT_TYPES = ("off_beat", "vocal_clash", "lufs_jump", "clipping", "bass_clash")


def synthesize_electronic_track(
    *,
    bpm: float = 124.0,
    duration_sec: float = 45.0,
    sr: int = 22050,
    root_hz: float = 55.0,
    seed: int = 0,
) -> tuple[np.ndarray, AudioTrackCard]:
    """Generate a clean 4-on-the-floor electronic track (kick + sub-bass + hi-hat) for QA tests."""
    rng = np.random.default_rng(seed)
    n = int(round(duration_sec * sr))
    t = np.arange(n, dtype=np.float64) / float(sr)
    beat_dur = 60.0 / bpm
    beat_times: list[float] = []

    y = np.zeros(n, dtype=np.float64)
    k_len = int(round(0.16 * sr))
    kt = np.arange(k_len, dtype=np.float64) / float(sr)
    # Punchy pitch-swept kick drum (140 Hz -> 48 Hz)
    kick_freq = 48.0 + 92.0 * np.exp(-kt * 28.0)
    kick = 0.72 * np.sin(2.0 * np.pi * np.cumsum(kick_freq) / sr) * np.exp(-kt * 14.0)

    # Crisp off-beat hi-hat
    h_len = int(round(0.05 * sr))
    ht = np.arange(h_len, dtype=np.float64) / float(sr)
    hat = 0.14 * rng.standard_normal(h_len) * np.exp(-ht * 45.0)

    b_idx = 0
    while b_idx * beat_dur < duration_sec - 0.25:
        bt = b_idx * beat_dur
        beat_times.append(round(bt, 4))
        s = int(round(bt * sr))
        end = min(n, s + k_len)
        y[s:end] += kick[: end - s]

        hs = int(round((bt + 0.5 * beat_dur) * sr))
        hend = min(n, hs + h_len)
        if hs < n:
            y[hs:hend] += hat[: hend - hs]
        b_idx += 1

    # Subtle sub-bass pulse + airy high pad (leaves mid vocal band clean unless injected)
    sub = 0.20 * np.sin(2.0 * np.pi * root_hz * t)
    air = 0.05 * np.sin(2.0 * np.pi * 5280.0 * t)
    y = (y + sub + air).astype(np.float32)
    y = y / max(float(np.max(np.abs(y))), 1e-6) * 0.85

    card = AudioTrackCard(
        track_id=f"synth_{int(bpm)}_{seed}",
        audio_path="",
        sample_rate=sr,
        duration_sec=duration_sec,
        bpm=bpm,
        bpm_stability=1.0,
        beat_times=beat_times,
        phrase_boundaries_sec=[beat_times[i] for i in range(0, len(beat_times), 32)],
        key="A minor",
        camelot="8A",
        key_confidence=0.9,
        integrated_lufs=-14.0,
        peak_dbfs=-1.4,
        energy_curve=[0.8] * int(duration_sec),
        vocal_activity_mask=[False] * int(duration_sec),
    )
    return y, card


def _inject_vocal_signal(stem: np.ndarray, sr: int, freq_hz: float) -> np.ndarray:
    """Inject a tonal vocal-band (formant-like 600-1800 Hz) melody into a stem."""
    t = np.arange(len(stem), dtype=np.float64) / float(sr)
    vocal = (
        0.55 * np.sin(2.0 * np.pi * freq_hz * t)
        + 0.30 * np.sin(2.0 * np.pi * (freq_hz * 1.5) * t)
        + 0.20 * np.sin(2.0 * np.pi * (freq_hz * 2.0) * t)
    )
    return (stem.astype(np.float64) + vocal).astype(np.float32)


def _apply_defect(
    defect: str,
    card_a: AudioTrackCard,
    card_b: AudioTrackCard,
    spec: RenderSpec,
    clean_render: RenderResult,
    y_a: np.ndarray,
    y_b: np.ndarray,
    sr: int,
) -> tuple[RenderResult, RenderSpec]:
    """Produce a transition with one known ground-truth defect injected."""
    if defect == "off_beat":
        bad_spec = spec.model_copy(update={"inject_beat_shift_ms": 165.0})
        return render_transition(card_a, card_b, bad_spec, sr=sr, y_a=y_a, y_b=y_b), bad_spec

    if defect == "bass_clash":
        bad_spec = spec.model_copy(update={"enable_bass_swap": False})
        return render_transition(card_a, card_b, bad_spec, sr=sr, y_a=y_a, y_b=y_b), bad_spec

    if defect == "vocal_clash":
        y_a_v = _inject_vocal_signal(y_a, sr, 660.0)
        y_b_v = _inject_vocal_signal(y_b, sr, 880.0)
        return render_transition(card_a, card_b, spec, sr=sr, y_a=y_a_v, y_b=y_b_v), spec

    if defect == "lufs_jump":
        bad_mix = clean_render.mix.copy()
        s0, s1 = clean_render.overlap_start_sample, clean_render.overlap_end_sample
        bad_mix[s0:s1] *= 0.35  # ~-9 dB loudness dip during transition
        return replace(clean_render, mix=bad_mix), spec

    if defect == "clipping":
        bad_mix = clean_render.mix.copy()
        s0, s1 = clean_render.overlap_start_sample, clean_render.overlap_end_sample
        bad_mix[s0:s1] = np.clip(bad_mix[s0:s1] * 3.5, -1.0, 1.0)
        return replace(clean_render, mix=bad_mix), spec

    raise ValueError(f"Unknown defect type: {defect}")


def _render_gym_markdown(report: dict) -> str:
    lines = [
        "# AI DJ — Phase 2 Defect Gym Calibration Report",
        "",
        f"- Clean transitions evaluated: **{report['clean_total']}** "
        f"(false-positive rate: **{report['clean_false_positive_rate'] * 100:.1f}%**)",
        f"- Injected defect trials: **{report['defect_total']}** "
        f"(overall detection recall: **{report['overall_defect_recall'] * 100:.1f}%**)",
        f"- Phase 2 Exit Gate (>95% recall): "
        f"**{'PASS' if report['passed_exit_gate'] else 'FAIL'}**",
        "",
        "| Defect Type | Trials | Detected | Recall |",
        "| --- | ---: | ---: | ---: |",
    ]
    for defect, stats in report["per_defect"].items():
        lines.append(
            f"| `{defect}` | {stats['trials']} | {stats['detected']} "
            f"| {stats['recall'] * 100:.1f}% |"
        )
    lines.append("")
    return "\n".join(lines)


def run_defect_gym(
    out_dir: Path,
    *,
    num_pairs: int = 8,
    sr: int = 22050,
    thresholds: QAThresholds | None = None,
    cc_audio_dir: Path | None = None,
) -> dict:
    """Run the Defect Gym across `num_pairs` clean transitions x 5 injected defect types."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = thresholds or QAThresholds()

    clean_fp = 0
    clean_total = 0
    per_defect = {d: {"trials": 0, "detected": 0, "recall": 0.0} for d in DEFECT_TYPES}

    bpms = [120.0, 122.0, 124.0, 126.0, 128.0, 130.0, 132.0, 125.0]
    roots = [55.0, 61.74, 65.41, 73.42, 82.41, 49.0, 58.27, 69.30]

    for i in range(num_pairs):
        bpm_a = bpms[i % len(bpms)]
        bpm_b = bpm_a + (1.5 if i % 2 == 0 else -1.5)
        y_a, card_a = synthesize_electronic_track(
            bpm=bpm_a, duration_sec=42.0, sr=sr, root_hz=roots[i % len(roots)], seed=100 + i
        )
        y_b, card_b = synthesize_electronic_track(
            bpm=bpm_b, duration_sec=42.0, sr=sr, root_hz=roots[(i + 1) % len(roots)], seed=200 + i
        )
        spec = RenderSpec(
            cue_out_beat_a=32,
            cue_in_beat_b=16,
            overlap_beats=16,
            transition_style="bass_swap",
            context_beats=12,
        )
        clean_res = render_transition(card_a, card_b, spec, sr=sr, y_a=y_a, y_b=y_b)
        clean_qa = verify_transition(clean_res, card_a, card_b, spec, thresholds=cfg)
        clean_total += 1
        if not clean_qa.passed:
            clean_fp += 1

        for defect in DEFECT_TYPES:
            bad_res, bad_spec = _apply_defect(
                defect, card_a, card_b, spec, clean_res, y_a, y_b, sr
            )
            bad_qa = verify_transition(bad_res, card_a, card_b, bad_spec, thresholds=cfg)
            per_defect[defect]["trials"] += 1
            if defect in bad_qa.defects_detected:
                per_defect[defect]["detected"] += 1

    # Optional smoke render on a real CC audio pair if present
    cc_sample_report: dict | None = None
    if cc_audio_dir and (cc_audio_dir / "manifest.json").exists():
        manifest = json.loads((cc_audio_dir / "manifest.json").read_text(encoding="utf-8"))
        if len(manifest) >= 2:
            c_a = analyze_track(Path(manifest[0]["audio_path"]), track_id=manifest[0]["track_id"])
            c_b = analyze_track(Path(manifest[1]["audio_path"]), track_id=manifest[1]["track_id"])
            cc_spec = RenderSpec(
                cue_out_beat_a=min(32, max(8, (len(c_a.beat_times) // 16) * 8)),
                cue_in_beat_b=min(16, max(0, (len(c_b.beat_times) // 16) * 8)),
                overlap_beats=16,
                transition_style="bass_swap",
                context_beats=8,
            )
            sample_wav = out_dir / "sample_cc_transition.wav"
            cc_render = render_transition(c_a, c_b, cc_spec, out_wav=sample_wav, sr=sr)
            cc_qa = verify_transition(cc_render, c_a, c_b, cc_spec, thresholds=cfg)
            cc_sample_report = {
                "track_a": c_a.track_id,
                "track_b": c_b.track_id,
                "bpm_a": c_a.bpm,
                "bpm_b": c_b.bpm,
                "camelot_a": c_a.camelot,
                "camelot_b": c_b.camelot,
                "wav_path": str(sample_wav),
                "qa": cc_qa.model_dump(),
            }

    total_trials = sum(v["trials"] for v in per_defect.values())
    total_detected = sum(v["detected"] for v in per_defect.values())
    for stats in per_defect.values():
        stats["recall"] = round(stats["detected"] / max(1, stats["trials"]), 4)

    overall_recall = round(total_detected / max(1, total_trials), 4)
    fp_rate = round(clean_fp / max(1, clean_total), 4)

    report = {
        "clean_total": clean_total,
        "clean_false_positives": clean_fp,
        "clean_false_positive_rate": fp_rate,
        "defect_total": total_trials,
        "defect_detected": total_detected,
        "overall_defect_recall": overall_recall,
        "passed_exit_gate": overall_recall >= 0.95 and fp_rate <= 0.10,
        "per_defect": per_defect,
        "cc_sample_render": cc_sample_report,
    }

    (out_dir / "defect_gym_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "defect_gym_report.md").write_text(_render_gym_markdown(report), encoding="utf-8")
    return report
