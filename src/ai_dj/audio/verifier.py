"""Automated technical QA verifier and verifiable RL reward for rendered DJ transitions."""

from __future__ import annotations

import librosa
import numpy as np
import pyloudnorm as pyln
from pydantic import BaseModel, Field
from scipy.signal import butter, sosfiltfilt

from ai_dj.audio.analyzer import AudioTrackCard, camelot_distance, estimate_vocal_activity_mask
from ai_dj.audio.critic import PsychoacousticCritique, evaluate_transition_psychoacoustics
from ai_dj.audio.renderer import RenderResult, RenderSpec


class QAThresholds(BaseModel):
    """Calibration thresholds for deterministic transition QA defect detection."""

    max_beat_drift_ms: float = 45.0
    max_vocal_clash_ratio: float = 0.25
    max_lufs_jump_db: float = 3.5
    max_peak_dbfs: float = -0.15
    max_clipped_samples: int = 8
    max_low_band_excess_db: float = 1.6
    max_drum_clash_ratio: float = 1.38
    max_step_rms_db: float = 4.5
    max_tempo_stretch_pct: float = 8.0


class TransitionQAReport(BaseModel):
    """Structured technical QA measurements, defect flags, and RL reward for a rendered mix."""

    beat_drift_ms: float
    beat_aligned: bool
    phrase_aligned: bool
    vocal_clash_ratio: float
    has_vocal_clash: bool
    lufs_pre: float
    lufs_overlap: float
    lufs_post: float
    lufs_jump_db: float
    has_lufs_jump: bool
    peak_dbfs: float
    clipped_samples: int
    has_clipping: bool
    low_band_excess_db: float
    has_bass_clash: bool
    drum_clash_ratio: float = 1.0
    has_drum_clash: bool = False
    max_step_rms_db: float = 0.0
    has_sudden_jump: bool = False
    tempo_stretch_pct: float
    excessive_stretch: bool
    camelot_dist: int
    psychoacoustic: PsychoacousticCritique | None = None
    defects_detected: list[str] = Field(default_factory=list)
    passed: bool
    verifiable_reward: float


def _region_lufs(y: np.ndarray, sr: int) -> float:
    meter = pyln.Meter(sr)
    x = y.astype(np.float64)
    n_samples = x.shape[-1]
    if n_samples >= sr:
        meas_in = x.T if x.ndim == 2 else x
        val = float(meter.integrated_loudness(meas_in))
        if np.isfinite(val):
            return round(val, 2)
    rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
    return round(20.0 * np.log10(rms), 2)


def _measure_beat_drift_ms(
    stem_a: np.ndarray,
    stem_b: np.ndarray,
    sr: int,
    bpm: float,
) -> float:
    """Measure beat/onset phase offset (ms) between Track A and Track B inside the overlap."""
    hop = 256
    env_a = librosa.onset.onset_strength(y=stem_a, sr=sr, hop_length=hop)
    env_b = librosa.onset.onset_strength(y=stem_b, sr=sr, hop_length=hop)
    n = min(len(env_a), len(env_b))
    if n < 16:
        return 0.0
    a = env_a[:n] - np.mean(env_a[:n])
    b = env_b[:n] - np.mean(env_b[:n])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.0

    # Search lag within +/- half a beat period
    half_beat_sec = 0.5 * (60.0 / max(bpm, 60.0))
    max_lag_frames = max(2, int(round(half_beat_sec * sr / hop)))
    best_lag = 0
    best_corr = -1e9
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        if lag >= 0:
            corr = float(np.dot(a[lag:], b[: n - lag]))
        else:
            corr = float(np.dot(a[: n + lag], b[-lag:]))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag

    drift_ms = abs(best_lag) * (hop / float(sr)) * 1000.0
    return round(drift_ms, 2)


def _low_band_rms_db(y: np.ndarray, sr: int, cutoff_hz: float = 250.0) -> float:
    if y.shape[-1] < 64:
        return -60.0
    sos = butter(4, cutoff_hz / (0.5 * sr), btype="lowpass", output="sos")
    low = sosfiltfilt(sos, y.astype(np.float64), axis=-1)
    rms = float(np.sqrt(np.mean(low**2)) + 1e-9)
    return 20.0 * np.log10(rms)


def verify_transition(
    render: RenderResult,
    card_a: AudioTrackCard,
    card_b: AudioTrackCard,
    spec: RenderSpec,
    *,
    thresholds: QAThresholds | None = None,
    include_psychoacoustic: bool = False,
) -> TransitionQAReport:
    """Run deterministic technical QA checks on a rendered transition."""
    cfg = thresholds or QAThresholds()
    sr = render.sample_rate
    raw_mix = render.mix
    s0, s1 = render.overlap_start_sample, render.overlap_end_sample

    pre_seg = raw_mix[..., :s0]
    ov_seg = raw_mix[..., s0:s1]
    post_seg = raw_mix[..., s1:]

    # 1. Beat alignment drift inside overlap
    beat_drift_ms = _measure_beat_drift_ms(
        render.track_a_overlap_stem,
        render.track_b_overlap_stem,
        sr,
        card_a.bpm,
    )
    beat_aligned = beat_drift_ms <= cfg.max_beat_drift_ms

    # 2. Phrase-grid alignment (bar/2-bar boundary)
    phrase_aligned = (spec.cue_out_beat_a % 8 == 0) and (spec.cue_in_beat_b % 8 == 0)

    # 3. Vocal-on-vocal clash ratio during overlap (0.5s frame resolution)
    stem_a_v = (
        render.rendered_a_overlap_stem
        if render.rendered_a_overlap_stem is not None
        else render.track_a_overlap_stem
    )
    stem_b_v = (
        render.rendered_b_overlap_stem
        if render.rendered_b_overlap_stem is not None
        else render.track_b_overlap_stem
    )
    if spec.transition_style == "quick_cut" and card_a.bpm > 0:
        cut_n = min(
            len(stem_a_v), len(stem_b_v), max(256, int(round(2.0 * (60.0 / card_a.bpm) * sr)))
        )
        stem_a_v = stem_a_v[:cut_n]
        stem_b_v = stem_b_v[:cut_n]
    mask_a = estimate_vocal_activity_mask(stem_a_v, sr, hop_sec=0.5)
    mask_b = estimate_vocal_activity_mask(stem_b_v, sr, hop_sec=0.5)
    n_v = min(len(mask_a), len(mask_b))
    if n_v > 0:
        clash_frames = sum(1 for i in range(n_v) if mask_a[i] and mask_b[i])
        vocal_clash_ratio = round(clash_frames / float(n_v), 4)
    else:
        vocal_clash_ratio = 0.0
    has_vocal_clash = vocal_clash_ratio > cfg.max_vocal_clash_ratio

    # 4. Loudness continuity (pre vs overlap vs post)
    lufs_pre = _region_lufs(pre_seg, sr)
    lufs_ov = _region_lufs(ov_seg, sr)
    lufs_post = _region_lufs(post_seg, sr)
    lufs_jump_db = round(max(abs(lufs_ov - lufs_pre), abs(lufs_post - lufs_ov)), 2)
    has_lufs_jump = lufs_jump_db > cfg.max_lufs_jump_db

    # 5. Digital clipping check (across all channels)
    peak = float(np.max(np.abs(raw_mix))) if raw_mix.size else 0.0
    peak_dbfs = round(20.0 * np.log10(max(peak, 1e-9)), 2)
    clipped_samples = int(np.sum(np.abs(raw_mix) >= 0.985))
    has_clipping = (peak_dbfs > cfg.max_peak_dbfs) or (clipped_samples >= cfg.max_clipped_samples)

    # 6. Low-band (<250 Hz) bass-clash excess during overlap
    low_pre = _low_band_rms_db(pre_seg, sr)
    low_ov = _low_band_rms_db(ov_seg, sr)
    low_post = _low_band_rms_db(post_seg, sr)
    low_ref = max(low_pre, low_post)
    low_band_excess_db = round(low_ov - low_ref, 2)
    has_bass_clash = low_band_excess_db > cfg.max_low_band_excess_db

    # 6b. Dual-drum / transient clash ratio (onset flux in overlap vs max of solo pre/post)
    def _mean_onset(sig: np.ndarray) -> float:
        m = np.mean(sig, axis=0) if sig.ndim == 2 else sig
        if len(m) < 1024:
            return 0.0
        return float(np.mean(librosa.onset.onset_strength(y=m.astype(np.float32), sr=sr)))

    ons_pre = _mean_onset(pre_seg)
    ons_post = _mean_onset(post_seg)
    ons_a_ov = _mean_onset(render.track_a_overlap_stem)
    ons_b_ov = _mean_onset(render.track_b_overlap_stem)
    ons_ov = _mean_onset(ov_seg)
    ons_ref = max(ons_pre, ons_post, ons_a_ov, ons_b_ov, 1e-6)
    drum_clash_ratio = round(ons_ov / ons_ref, 3)
    has_drum_clash = (
        spec.transition_style != "quick_cut" and drum_clash_ratio > cfg.max_drum_clash_ratio
    )

    # 6c. Local 0.5s loudness discontinuity inside overlap (catches sudden cuts/jumps)
    mono_mix = np.mean(raw_mix, axis=0) if raw_mix.ndim == 2 else raw_mix
    win_n = max(512, int(0.5 * sr))
    step_start = max(0, s0 - win_n)
    step_end = min(len(mono_mix), s1 + win_n)
    ov_region = mono_mix[step_start:step_end]
    rms_steps: list[float] = []
    for pos in range(0, max(0, len(ov_region) - win_n + 1), win_n):
        seg_w = ov_region[pos : pos + win_n]
        rms_steps.append(20.0 * float(np.log10(np.sqrt(np.mean(seg_w**2)) + 1e-9)))
    max_step_rms_db = (
        round(max(abs(rms_steps[i] - rms_steps[i - 1]) for i in range(1, len(rms_steps))), 2)
        if len(rms_steps) >= 2
        else 0.0
    )
    has_sudden_jump = (
        spec.transition_style != "quick_cut" and max_step_rms_db > cfg.max_step_rms_db
    )

    # 7. Tempo stretch & harmonic compatibility
    excessive_stretch = render.tempo_stretch_pct > cfg.max_tempo_stretch_pct
    c_dist = camelot_distance(card_a.camelot, card_b.camelot)

    # 7b. Psychoacoustic Mix Critic (Plomp-Levelt, cross-chroma, peak dual-drum, Bark masking)
    psycho: PsychoacousticCritique | None = None
    if include_psychoacoustic or render.rendered_a_drums is not None:
        psycho = evaluate_transition_psychoacoustics(
            ov_seg,
            stem_a_v,
            stem_b_v,
            sr,
            overlap_start_sec=s0 / float(sr),
            rendered_a_drums=render.rendered_a_drums,
            rendered_b_drums=render.rendered_b_drums,
            rendered_a_harm=render.rendered_a_harm,
            rendered_b_harm=render.rendered_b_harm,
            solo_a_ref=pre_seg,
            solo_b_ref=post_seg,
        )

    defects: list[str] = []
    if not beat_aligned:
        defects.append("off_beat")
    if has_vocal_clash:
        defects.append("vocal_clash")
    if has_lufs_jump:
        defects.append("lufs_jump")
    if has_clipping:
        defects.append("clipping")
    if has_bass_clash:
        defects.append("bass_clash")
    if has_drum_clash:
        defects.append("drum_clash")
    if has_sudden_jump:
        defects.append("sudden_jump")
    if excessive_stretch:
        defects.append("excessive_stretch")
    if psycho is not None:
        for issue in psycho.issues:
            if issue in ("harmonic_clash", "dual_drum_overlap") and issue not in defects:
                defects.append(issue)

    # Verifiable 0..1 RL reward (hard-gated on technical defects)
    reward = 1.0
    if not beat_aligned:
        reward -= 0.35
    if has_vocal_clash:
        reward -= 0.25
    if has_lufs_jump:
        reward -= 0.20
    if has_clipping:
        reward -= 0.25
    if has_bass_clash:
        reward -= 0.20
    if has_drum_clash:
        reward -= 0.15
    if has_sudden_jump:
        reward -= 0.20
    if not phrase_aligned:
        reward -= 0.10
    if excessive_stretch:
        reward -= 0.20
    if psycho is not None:
        if "harmonic_clash" in psycho.issues:
            reward -= 0.20
        if "dual_drum_overlap" in psycho.issues:
            reward -= 0.20
    # Smooth bonus/penalty for harmonic proximity
    if c_dist > 2:
        reward -= min(0.10, 0.02 * (c_dist - 2))
    reward = round(max(0.0, min(1.0, reward)), 4)

    return TransitionQAReport(
        beat_drift_ms=beat_drift_ms,
        beat_aligned=beat_aligned,
        phrase_aligned=phrase_aligned,
        vocal_clash_ratio=vocal_clash_ratio,
        has_vocal_clash=has_vocal_clash,
        lufs_pre=lufs_pre,
        lufs_overlap=lufs_ov,
        lufs_post=lufs_post,
        lufs_jump_db=lufs_jump_db,
        has_lufs_jump=has_lufs_jump,
        peak_dbfs=peak_dbfs,
        clipped_samples=clipped_samples,
        has_clipping=has_clipping,
        low_band_excess_db=low_band_excess_db,
        has_bass_clash=has_bass_clash,
        drum_clash_ratio=drum_clash_ratio,
        has_drum_clash=has_drum_clash,
        max_step_rms_db=max_step_rms_db,
        has_sudden_jump=has_sudden_jump,
        tempo_stretch_pct=render.tempo_stretch_pct,
        excessive_stretch=excessive_stretch,
        camelot_dist=c_dist,
        psychoacoustic=psycho,
        defects_detected=defects,
        passed=len(defects) == 0,
        verifiable_reward=reward,
    )
