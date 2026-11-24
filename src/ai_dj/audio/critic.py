"""Psychoacoustic Mix Critic: per-second roughness, cross-chroma clash, dual-drum overlap,
and Bark-band masking with actionable DJ/DSP fix diagnostics.
"""

from __future__ import annotations

import librosa
import numpy as np
from pydantic import BaseModel, Field
from scipy.signal import find_peaks

# 12x12 semitone-interval dissonance weights (0 = consonant, 1 = harsh clash)
# Intervals: 0:unison, 1:m2, 2:M2, 3:m3, 4:M3, 5:P4, 6:tritone, 7:P5, 8:m6, 9:M6, 10:m7, 11:M7
_INTERVAL_DISSONANCE = np.array(
    [0.0, 1.0, 0.55, 0.10, 0.05, 0.05, 0.85, 0.0, 0.15, 0.10, 0.45, 0.95],
    dtype=np.float64,
)
_CHROMA_DISSONANCE_MAT = np.array(
    [[_INTERVAL_DISSONANCE[(i - j) % 12] for j in range(12)] for i in range(12)],
    dtype=np.float64,
)


class PsychoacousticCritique(BaseModel):
    """Per-second psychoacoustic critique & actionable diagnostics for a rendered transition."""

    roughness_excess: float = 0.0
    peak_roughness_sec: float = 0.0
    peak_harmonic_clash: float = 0.0
    mean_harmonic_clash: float = 0.0
    peak_harmonic_clash_sec: float = 0.0
    peak_dual_drum_overlap: float = 0.0
    dual_drum_duration_sec: float = 0.0
    peak_dual_drum_sec: float = 0.0
    peak_bark_masking: float = 0.0
    worst_masking_band: str = "none"
    peak_bark_masking_sec: float = 0.0
    naturalness_score: float = 1.0
    issues: list[str] = Field(default_factory=list)
    actionable_fixes: list[str] = Field(default_factory=list)


def _to_mono_f32(y: np.ndarray) -> np.ndarray:
    return np.mean(y, axis=0).astype(np.float32) if y.ndim == 2 else y.astype(np.float32)


def plomp_levelt_roughness(frame: np.ndarray, sr: int, *, top_k: int = 16) -> float:
    """Compute Sethares/Plomp-Levelt sensory roughness on the top spectral peaks of a frame."""
    if len(frame) < 512:
        return 0.0
    win = np.hanning(len(frame))
    spec = np.abs(np.fft.rfft(frame.astype(np.float64) * win))
    freqs = np.fft.rfftfreq(len(frame), d=1.0 / sr)
    # Focus on tonal/mid-high region (60 Hz .. 6000 Hz) where roughness is strongest
    mask = (freqs >= 60.0) & (freqs <= 6000.0)
    spec = spec[mask]
    freqs = freqs[mask]
    if spec.size < 4 or float(np.max(spec)) < 1e-8:
        return 0.0

    peaks, _ = find_peaks(spec, distance=2, height=float(np.max(spec)) * 0.08)
    if len(peaks) < 2:
        return 0.0
    if len(peaks) > top_k:
        top_idx = np.argsort(spec[peaks])[-top_k:]
        peaks = peaks[top_idx]

    p_freqs = freqs[peaks]
    p_amps = spec[peaks] / (float(np.sum(spec[peaks])) + 1e-9)

    # Vectorized pairwise Sethares (1993) roughness
    f_i = p_freqs[:, None]
    f_j = p_freqs[None, :]
    a_i = p_amps[:, None]
    a_j = p_amps[None, :]
    f_min = np.minimum(f_i, f_j)
    df = np.abs(f_i - f_j)
    s = 0.24 / (0.021 * f_min + 19.0)
    d = np.minimum(a_i, a_j) * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))
    # Upper triangle only (i < j)
    triu = np.triu_indices(len(peaks), k=1)
    return float(np.sum(np.maximum(0.0, d[triu])))


def evaluate_transition_psychoacoustics(
    ov_mix: np.ndarray,
    rendered_a: np.ndarray,
    rendered_b: np.ndarray,
    sr: int,
    *,
    overlap_start_sec: float = 0.0,
    rendered_a_drums: np.ndarray | None = None,
    rendered_b_drums: np.ndarray | None = None,
    rendered_a_harm: np.ndarray | None = None,
    rendered_b_harm: np.ndarray | None = None,
    solo_a_ref: np.ndarray | None = None,
    solo_b_ref: np.ndarray | None = None,
    hop_sec: float = 0.5,
) -> PsychoacousticCritique:
    """Evaluate a rendered transition overlap with 4 psychoacoustic metrics + actionable fixes."""
    mix_m = _to_mono_f32(ov_mix)
    a_m = _to_mono_f32(rendered_a)
    b_m = _to_mono_f32(rendered_b)
    n = min(len(mix_m), len(a_m), len(b_m))
    if n < int(sr * 0.5):
        return PsychoacousticCritique()
    mix_m, a_m, b_m = mix_m[:n], a_m[:n], b_m[:n]

    win_n = max(1024, int(round(hop_sec * sr)))
    n_frames = max(1, n // win_n)

    # 1. Plomp-Levelt sensory roughness excess (overlap vs solo track reference)
    ref_r = 0.0
    for ref in (solo_a_ref, solo_b_ref):
        if ref is not None and ref.size >= win_n:
            rm = _to_mono_f32(ref)
            ref_vals = [
                plomp_levelt_roughness(rm[i : i + win_n], sr)
                for i in range(0, min(len(rm) - win_n + 1, win_n * 8), win_n)
            ]
            if ref_vals:
                ref_r = max(ref_r, float(np.percentile(ref_vals, 85)))
    ov_rough = [
        plomp_levelt_roughness(mix_m[i * win_n : (i + 1) * win_n], sr) for i in range(n_frames)
    ]
    max_r_idx = int(np.argmax(ov_rough))
    roughness_excess = round(max(0.0, ov_rough[max_r_idx] - ref_r), 4)
    peak_roughness_sec = round(overlap_start_sec + max_r_idx * hop_sec, 2)

    # 2. Cross-chroma harmonic clash on rendered harmonic/melodic contributions
    ha = _to_mono_f32(rendered_a_harm)[:n] if rendered_a_harm is not None else a_m
    hb = _to_mono_f32(rendered_b_harm)[:n] if rendered_b_harm is not None else b_m
    harm_floor = max(float(np.sqrt(np.mean(ha**2))), float(np.sqrt(np.mean(hb**2))), 1e-4) * 0.25

    harm_clashes: list[float] = []
    for i in range(n_frames):
        s0, s1 = i * win_n, (i + 1) * win_n
        fa, fb = ha[s0:s1], hb[s0:s1]
        # Scale against local frame mix RMS so quiet breaks expose clashes while
        # quiet background synth teases under loud drums don't false-trigger.
        local_ref = max(float(np.sqrt(np.mean(mix_m[s0:s1] ** 2))) * 0.6, harm_floor, 1e-4)
        wa = min(1.0, float(np.sqrt(np.mean(fa**2))) / (local_ref + 1e-9))
        wb = min(1.0, float(np.sqrt(np.mean(fb**2))) / (local_ref + 1e-9))
        co_energy = min(wa, wb)
        if co_energy < 0.08:
            harm_clashes.append(0.0)
            continue
        ca = np.mean(librosa.feature.chroma_stft(y=fa, sr=sr), axis=1)
        cb = np.mean(librosa.feature.chroma_stft(y=fb, sr=sr), axis=1)
        ca = ca / (float(np.sum(ca)) + 1e-9)
        cb = cb / (float(np.sum(cb)) + 1e-9)
        raw_diss = float(ca @ _CHROMA_DISSONANCE_MAT @ cb)
        harm_clashes.append(co_energy * raw_diss)

    peak_hc_idx = int(np.argmax(harm_clashes))
    peak_harmonic_clash = round(float(harm_clashes[peak_hc_idx]), 4)
    mean_harmonic_clash = round(float(np.mean(harm_clashes)), 4)
    peak_harmonic_clash_sec = round(overlap_start_sec + peak_hc_idx * hop_sec, 2)

    # 3. Peak per-frame dual-drum overlap (measures when BOTH drum kits play simultaneously)
    if rendered_a_drums is not None and rendered_b_drums is not None:
        da = _to_mono_f32(rendered_a_drums)[:n]
        db = _to_mono_f32(rendered_b_drums)[:n]
    else:
        da = librosa.effects.percussive(a_m)
        db = librosa.effects.percussive(b_m)
    da_ref = max(float(np.percentile(np.abs(da), 95)) * 0.35, 1e-4)
    db_ref = max(float(np.percentile(np.abs(db), 95)) * 0.35, 1e-4)
    dual_drums: list[float] = []
    for i in range(n_frames):
        s0, s1 = i * win_n, (i + 1) * win_n
        ra = min(1.0, float(np.sqrt(np.mean(da[s0:s1] ** 2))) / da_ref)
        rb = min(1.0, float(np.sqrt(np.mean(db[s0:s1] ** 2))) / db_ref)
        dual_drums.append(min(ra, rb))
    peak_dd_idx = int(np.argmax(dual_drums))
    peak_dual_drum_overlap = round(float(dual_drums[peak_dd_idx]), 4)
    dual_drum_duration_sec = round(sum(hop_sec for v in dual_drums if v >= 0.40), 2)
    peak_dual_drum_sec = round(overlap_start_sec + peak_dd_idx * hop_sec, 2)

    # 4. 24-band Mel/Bark critical-band masking overlap
    mel_a = librosa.feature.melspectrogram(y=a_m, sr=sr, n_mels=24, hop_length=win_n)
    mel_b = librosa.feature.melspectrogram(y=b_m, sr=sr, n_mels=24, hop_length=win_n)
    nf_m = min(mel_a.shape[1], mel_b.shape[1])
    band_names = [
        ("low", 0, 5),
        ("low_mid", 5, 11),
        ("high_mid", 11, 18),
        ("air", 18, 24),
    ]
    frame_masking: list[float] = []
    band_peaks = {name: 0.0 for name, _, _ in band_names}
    for f in range(nf_m):
        ea = mel_a[:, f]
        eb = mel_b[:, f]
        denom = float(np.sum(np.maximum(ea, eb))) + 1e-9
        frame_masking.append(float(np.sum(np.minimum(ea, eb))) / denom)
        for name, b0, b1 in band_names:
            bd = float(np.sum(np.maximum(ea[b0:b1], eb[b0:b1]))) + 1e-9
            bm = float(np.sum(np.minimum(ea[b0:b1], eb[b0:b1]))) / bd
            if bm > band_peaks[name]:
                band_peaks[name] = bm
    peak_bm_idx = int(np.argmax(frame_masking)) if frame_masking else 0
    peak_bark_masking = round(float(frame_masking[peak_bm_idx]) if frame_masking else 0.0, 4)
    peak_bark_masking_sec = round(overlap_start_sec + peak_bm_idx * hop_sec, 2)
    worst_masking_band = max(band_peaks, key=lambda k: band_peaks[k]) if band_peaks else "none"

    # Summarize issues & actionable DJ/DSP fixes
    issues: list[str] = []
    fixes: list[str] = []
    score = 1.0

    if peak_harmonic_clash >= 0.12:
        issues.append("harmonic_clash")
        fixes.append(
            f"strip_melody_a_before_b: cross-chroma clash={peak_harmonic_clash:.2f} "
            f"@ {peak_harmonic_clash_sec:.1f}s (fade Track A other/bass out before Track B enters)"
        )
        score -= min(0.35, peak_harmonic_clash * 1.2)
    if peak_dual_drum_overlap >= 0.55 and dual_drum_duration_sec >= 1.5:
        issues.append("dual_drum_overlap")
        fixes.append(
            f"isolate_single_drum_kit: dual_drum={peak_dual_drum_overlap:.2f} for "
            f"{dual_drum_duration_sec:.1f}s around {peak_dual_drum_sec:.1f}s "
            "(do not overlap both full drum kits)"
        )
        score -= min(0.35, 0.20 + 0.04 * dual_drum_duration_sec)
    if roughness_excess >= 0.06:
        issues.append("sensory_roughness")
        fixes.append(
            f"reduce_roughness: Plomp-Levelt excess={roughness_excess:.3f} "
            f"@ {peak_roughness_sec:.1f}s"
        )
        score -= min(0.20, roughness_excess * 2.0)
    if peak_bark_masking >= 0.45:
        issues.append(f"masking_{worst_masking_band}")
        fixes.append(
            f"carve_{worst_masking_band}: Bark masking={peak_bark_masking:.2f} "
            f"@ {peak_bark_masking_sec:.1f}s"
        )
        score -= min(0.15, (peak_bark_masking - 0.40) * 0.6)

    return PsychoacousticCritique(
        roughness_excess=roughness_excess,
        peak_roughness_sec=peak_roughness_sec,
        peak_harmonic_clash=peak_harmonic_clash,
        mean_harmonic_clash=mean_harmonic_clash,
        peak_harmonic_clash_sec=peak_harmonic_clash_sec,
        peak_dual_drum_overlap=peak_dual_drum_overlap,
        dual_drum_duration_sec=dual_drum_duration_sec,
        peak_dual_drum_sec=peak_dual_drum_sec,
        peak_bark_masking=peak_bark_masking,
        worst_masking_band=worst_masking_band,
        peak_bark_masking_sec=peak_bark_masking_sec,
        naturalness_score=round(max(0.0, min(1.0, score)), 4),
        issues=issues,
        actionable_fixes=fixes,
    )
