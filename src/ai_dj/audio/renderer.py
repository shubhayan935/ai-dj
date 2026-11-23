"""Deterministic MixPlan -> WAV DSP renderer (44.1 kHz stereo, WSOLA stretch, phase-lock, DJ EQ)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pydantic import BaseModel, Field
from scipy.signal import butter, sosfiltfilt

from ai_dj.audio.analyzer import AudioTrackCard, load_audio_mono, load_audio_stereo

RenderStyle = Literal[
    "quick_cut",
    "standard_blend",
    "long_blend",
    "bass_swap",
    "filter_sweep",
    "stem_strip_blend",
]


class RenderSpec(BaseModel):
    """Execution parameters for rendering a single Track A -> Track B DJ transition."""

    cue_out_beat_a: int = Field(ge=0)
    cue_in_beat_b: int = Field(ge=0)
    overlap_beats: int = Field(default=32, ge=4, le=128)
    transition_style: RenderStyle = "bass_swap"
    bass_swap_beat_frac: float = Field(default=0.5, ge=0.1, le=0.9)
    context_beats: int = Field(default=16, ge=4, le=32)
    target_lufs: float = -14.0
    # Defect-injection hooks (used by the Defect Gym; always 0 / True in normal rendering)
    inject_beat_shift_ms: float = 0.0
    enable_bass_swap: bool = True
    enable_limiter: bool = True


@dataclass
class RenderResult:
    """Rendered audio arrays + exact sample boundaries for downstream QA verification."""

    mix: np.ndarray  # (N,) mono or (2, N) stereo
    track_a_overlap_stem: np.ndarray  # mono overlap stem for QA
    track_b_overlap_stem: np.ndarray  # mono overlap stem for QA
    sample_rate: int
    overlap_start_sample: int
    overlap_end_sample: int
    tempo_stretch_pct: float
    phase_correction_ms: float = 0.0
    out_wav: Path | None = None


def _to_mono(y: np.ndarray) -> np.ndarray:
    return y.mean(axis=0).astype(np.float32) if y.ndim == 2 else y.astype(np.float32)


def _ensure_2d(y: np.ndarray) -> tuple[np.ndarray, bool]:
    if y.ndim == 1:
        return y[np.newaxis, :].astype(np.float32), False
    return y.astype(np.float32), True


def split_three_band(
    y: np.ndarray,
    sr: int,
    low_hz: float = 250.0,
    high_hz: float = 4000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero-phase 3-band EQ split (Low < 250 Hz, Mid 250-4000 Hz, High > 4000 Hz)."""
    x = y.astype(np.float64)
    nyq = 0.5 * sr
    sos_low = butter(4, low_hz / nyq, btype="lowpass", output="sos")
    sos_high = butter(4, min(high_hz / nyq, 0.95), btype="highpass", output="sos")
    low = sosfiltfilt(sos_low, x, axis=-1)
    high = sosfiltfilt(sos_high, x, axis=-1)
    mid = x - low - high
    return low, mid, high


def _apply_filter_sweep(
    y: np.ndarray,
    sr: int,
    *,
    btype: Literal["highpass", "lowpass"],
    start_hz: float,
    end_hz: float,
    n_steps: int = 12,
) -> np.ndarray:
    """Smooth DJ filter sweep via overlapping crossfaded SOS filterbank bands."""
    x = y.astype(np.float64)
    n_samples = x.shape[-1]
    if n_samples < 512:
        return x
    nyq = 0.5 * sr
    freqs = np.geomspace(
        max(30.0, min(start_hz, nyq * 0.92)),
        max(30.0, min(end_hz, nyq * 0.92)),
        n_steps,
    )
    out = np.zeros_like(x)
    t = np.linspace(0.0, n_steps - 1, n_samples, dtype=np.float64)
    for idx, fc in enumerate(freqs):
        # Triangular/cosine partition-of-unity weight for smooth sweep with zero zipper noise
        dist = np.abs(t - idx)
        w = np.where(dist < 1.0, 0.5 * (1.0 + np.cos(np.pi * dist)), 0.0)
        if not np.any(w > 0):
            continue
        sos = butter(2, fc / nyq, btype=btype, output="sos")
        filtered = sosfiltfilt(sos, x, axis=-1)
        out += filtered * w
    return out


def normalize_lufs(y: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """Loudness-normalize mono or stereo audio to `target_lufs` (ITU-R BS.1770)."""
    meter = pyln.Meter(sr)
    x = y.astype(np.float64)
    n_samples = x.shape[-1]
    if n_samples >= sr:
        meas_in = x.T if x.ndim == 2 else x
        cur = float(meter.integrated_loudness(meas_in))
    else:
        rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
        cur = 20.0 * np.log10(rms)
    if not np.isfinite(cur) or cur < -60.0:
        return y.astype(np.float32)
    gain_db = max(-18.0, min(18.0, target_lufs - cur))
    return (x * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def _soft_limit(y: np.ndarray, ceiling: float = 0.94) -> np.ndarray:
    """Gentle headroom soft-clipper at -0.54 dBFS to prevent digital clipping."""
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak <= ceiling:
        return y
    return (ceiling * np.tanh(y / ceiling)).astype(np.float32)


def _time_stretch_wsola(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Transient-preserving WSOLA time-stretch via ffmpeg `atempo` (preserves stereo)."""
    arr, is_stereo = _ensure_2d(y)
    n_ch = arr.shape[0]
    interleaved = arr.T.astype(np.float32).tobytes()
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        str(n_ch),
        "-i",
        "pipe:0",
        "-filter:a",
        f"atempo={rate:.6f}",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        str(n_ch),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, input=interleaved, check=True, capture_output=True)
        out_raw = np.frombuffer(proc.stdout, dtype=np.float32)
        n_frames = len(out_raw) // n_ch
        if n_frames > 0:
            out_2d = out_raw[: n_frames * n_ch].reshape(n_frames, n_ch).T.copy()
            return out_2d if is_stereo else out_2d[0]
    except Exception:
        pass
    # Fallback to librosa per channel
    stretched = np.vstack([librosa.effects.time_stretch(ch, rate=rate) for ch in arr])
    return stretched if is_stereo else stretched[0]


def _phase_align_shift_samples(
    ov_a_mono: np.ndarray,
    ov_b_mono: np.ndarray,
    sr: int,
    bpm: float,
) -> int:
    """Find sample shift in [-half_beat, +half_beat] that locks Track B's kicks to Track A."""
    hop = 256 if sr <= 24000 else 512
    env_a = librosa.onset.onset_strength(y=ov_a_mono, sr=sr, hop_length=hop)
    env_b = librosa.onset.onset_strength(y=ov_b_mono, sr=sr, hop_length=hop)
    n = min(len(env_a), len(env_b))
    if n < 16:
        return 0
    a = env_a[:n] - np.mean(env_a[:n])
    b = env_b[:n] - np.mean(env_b[:n])
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0

    half_beat_sec = 0.5 * (60.0 / max(bpm, 60.0))
    max_lag = max(2, int(round(half_beat_sec * sr / hop)))
    best_lag = 0
    best_corr = -1e9
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = float(np.dot(a[lag:], b[: n - lag]))
        else:
            corr = float(np.dot(a[: n + lag], b[-lag:]))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    # Positive lag means A peaks occur `lag` frames after B -> delay B by +lag * hop
    return int(best_lag * hop)


def _beat_time(card: AudioTrackCard, beat_idx: int) -> float:
    if card.beat_times and 0 <= beat_idx < len(card.beat_times):
        return card.beat_times[beat_idx]
    beat_dur = 60.0 / max(card.bpm, 60.0)
    return beat_idx * beat_dur


def _shift_signal(sig: np.ndarray, shift_samples: int, target_len: int) -> np.ndarray:
    out = np.zeros((sig.shape[0], target_len), dtype=np.float32)
    src_len = sig.shape[1]
    if shift_samples >= 0:
        avail = min(src_len, target_len - shift_samples)
        if avail > 0:
            out[:, shift_samples : shift_samples + avail] = sig[:, :avail]
    else:
        src_off = min(src_len, -shift_samples)
        avail = min(src_len - src_off, target_len)
        if avail > 0:
            out[:, :avail] = sig[:, src_off : src_off + avail]
    return out


def _smoothstep(x: np.ndarray) -> np.ndarray:
    c = np.clip(x, 0.0, 1.0)
    return c * c * (3.0 - 2.0 * c)


def render_transition(
    card_a: AudioTrackCard,
    card_b: AudioTrackCard,
    spec: RenderSpec,
    out_wav: Path | None = None,
    *,
    sr: int = 44100,
    stereo: bool = True,
    y_a: np.ndarray | None = None,
    y_b: np.ndarray | None = None,
) -> RenderResult:
    """Render a beatmatched, phase-locked, loudness-matched DJ transition from Track A to B."""
    if y_a is not None:
        audio_a_2d, is_stereo_a = _ensure_2d(y_a)
    elif stereo:
        audio_a_2d, is_stereo_a = load_audio_stereo(Path(card_a.audio_path), sr=sr), True
    else:
        audio_a_2d, is_stereo_a = _ensure_2d(load_audio_mono(Path(card_a.audio_path), sr=sr))

    if y_b is not None:
        audio_b_2d, _ = _ensure_2d(y_b)
    elif stereo:
        audio_b_2d = load_audio_stereo(Path(card_b.audio_path), sr=sr)
    else:
        audio_b_2d, _ = _ensure_2d(load_audio_mono(Path(card_b.audio_path), sr=sr))

    # Match channel counts if caller mixed mono/stereo arrays
    if audio_a_2d.shape[0] != audio_b_2d.shape[0]:
        if audio_a_2d.shape[0] == 2:
            audio_b_2d = np.vstack([audio_b_2d[0], audio_b_2d[0]])
        else:
            audio_a_2d = np.vstack([audio_a_2d[0], audio_a_2d[0]])
            is_stereo_a = True
    n_ch = audio_a_2d.shape[0]

    beat_dur_a = 60.0 / max(card_a.bpm, 60.0)
    beat_dur_b = 60.0 / max(card_b.bpm, 60.0)

    # Tempo-stretch rate to match Track B's BPM to Track A's BPM (octave-safe)
    rate = float(card_a.bpm) / max(float(card_b.bpm), 1.0)
    tempo_stretch_pct = round(abs(rate - 1.0) * 100.0, 3)

    t_cue_a = _beat_time(card_a, spec.cue_out_beat_a)
    t_cue_b = _beat_time(card_b, spec.cue_in_beat_b)

    pre_n = int(round(spec.context_beats * beat_dur_a * sr))
    ov_n = max(sr // 2, int(round(spec.overlap_beats * beat_dur_a * sr)))
    post_n = int(round(spec.context_beats * beat_dur_a * sr))
    ov_sec = ov_n / float(sr)

    # Extract Track A segment: [cue_out - pre, cue_out + overlap]
    a_cue_sample = int(round(t_cue_a * sr))
    a_start = max(0, a_cue_sample - pre_n)
    a_end = a_cue_sample + ov_n
    seg_a_full = np.zeros((n_ch, pre_n + ov_n), dtype=np.float32)
    src_a = audio_a_2d[:, a_start : min(audio_a_2d.shape[1], a_end)]
    offset_a = pre_n - (a_cue_sample - a_start)
    avail_a = min(src_a.shape[1], seg_a_full.shape[1] - offset_a)
    seg_a_full[:, offset_a : offset_a + avail_a] = src_a[:, :avail_a]
    seg_a_full = normalize_lufs(seg_a_full, sr, spec.target_lufs)

    # Extract Track B with a 1-beat lookbehind margin so post-stretch phase-lock can shift cleanly
    lead_in_orig = min(int(round(t_cue_b * sr)), int(round(beat_dur_b * sr)))
    b_start_orig = max(0, int(round(t_cue_b * sr)) - lead_in_orig)
    b_total_sec = (ov_sec + spec.context_beats * beat_dur_a) * rate + 3.0 * beat_dur_b
    b_needed_orig = int(round(b_total_sec * sr))
    raw_b = audio_b_2d[:, b_start_orig : min(audio_b_2d.shape[1], b_start_orig + b_needed_orig)]
    if raw_b.shape[1] < sr // 2:
        raw_b = np.pad(raw_b, ((0, 0), (0, max(0, sr - raw_b.shape[1]))))

    if abs(rate - 1.0) > 0.002 and 0.65 <= rate <= 1.50:
        stretched_b = _time_stretch_wsola(raw_b, sr, rate)
        applied_rate = rate
    else:
        stretched_b = raw_b.astype(np.float32)
        applied_rate = 1.0

    lead_in_stretched = int(round(lead_in_orig / max(applied_rate, 0.5)))
    base_b = _shift_signal(stretched_b, -lead_in_stretched, ov_n + post_n)

    # Post-stretch onset phase-lock (auto-aligns kicks to <15 ms unless defect shift is injected)
    phase_corr_samples = 0
    if spec.inject_beat_shift_ms == 0.0:
        ov_a_m = _to_mono(seg_a_full[:, pre_n : pre_n + ov_n])
        ov_b_m = _to_mono(base_b[:, :ov_n])
        phase_corr_samples = _phase_align_shift_samples(ov_a_m, ov_b_m, sr, card_a.bpm)
    inject_samples = int(round((spec.inject_beat_shift_ms / 1000.0) * sr))
    total_shift = -lead_in_stretched + phase_corr_samples + inject_samples
    seg_b_full = _shift_signal(stretched_b, total_shift, ov_n + post_n)
    seg_b_full = normalize_lufs(seg_b_full, sr, spec.target_lufs)

    # Split overlap segments into 3 zero-phase EQ bands
    ov_a = seg_a_full[:, pre_n : pre_n + ov_n]
    ov_b = seg_b_full[:, :ov_n]
    low_a, mid_a, high_a = split_three_band(ov_a, sr)
    low_b, mid_b, high_b = split_three_band(ov_b, sr)

    t = np.linspace(0.0, 1.0, ov_n, dtype=np.float64)
    swap_w = max(0.04, min(0.14, (2.0 * beat_dur_a) / max(ov_sec, 1e-3)))
    s_swap = _smoothstep((t - (spec.bass_swap_beat_frac - swap_w / 2.0)) / swap_w)

    if spec.enable_bass_swap:
        gain_a_low = np.cos(0.5 * np.pi * s_swap)
        gain_b_low = np.sin(0.5 * np.pi * s_swap)
    else:
        gain_a_low = np.ones(ov_n, dtype=np.float64)
        gain_b_low = np.ones(ov_n, dtype=np.float64)

    if spec.transition_style == "quick_cut":
        # Phrase-aligned 2-beat cut with a brief click-free high-pass exit tail on Track A
        cut_n = min(ov_n, max(256, int(round(2.0 * beat_dur_a * sr))))
        ramp = _smoothstep(np.linspace(0.0, 1.0, cut_n, dtype=np.float64))
        gain_a = np.zeros(ov_n, dtype=np.float64)
        gain_b = np.ones(ov_n, dtype=np.float64)
        gain_a[:cut_n] = np.cos(0.5 * np.pi * ramp)
        gain_b[:cut_n] = np.sin(0.5 * np.pi * ramp)
        ov_a_hp = _apply_filter_sweep(ov_a, sr, btype="highpass", start_hz=60.0, end_hz=1800.0)
        wet_in = _smoothstep(t / 0.05)
        ov_a_eff = ov_a * (1.0 - wet_in) + ov_a_hp * wet_in
        ov_mix = (ov_a_eff * gain_a + ov_b * gain_b).astype(np.float32)

    elif spec.transition_style == "filter_sweep":
        # Silky club filter transition: click-free wet/dry HPF out on A + LPF open on B + bass swap
        a_mh_dry = mid_a + high_a
        b_mh_dry = mid_b + high_b
        a_mh_filt = _apply_filter_sweep(
            a_mh_dry, sr, btype="highpass", start_hz=120.0, end_hz=2600.0
        )
        b_mh_filt = _apply_filter_sweep(
            b_mh_dry, sr, btype="lowpass", start_hz=600.0, end_hz=16000.0
        )
        wet_a = _smoothstep(t / 0.08)
        dry_b = _smoothstep((t - 0.90) / 0.10)
        a_mh = a_mh_dry * (1.0 - wet_a) + a_mh_filt * wet_a
        b_mh = b_mh_filt * (1.0 - dry_b) + b_mh_dry * dry_b
        g_a_mh = np.cos(0.5 * np.pi * _smoothstep((t - 0.15) / 0.85))
        g_b_mh = np.sin(0.5 * np.pi * _smoothstep(t / 0.80))
        ov_mix = (
            low_a * gain_a_low + low_b * gain_b_low + a_mh * g_a_mh + b_mh * g_b_mh
        ).astype(np.float32)

    elif spec.transition_style == "stem_strip_blend":
        # HPSS element stripping: strip Track A's melody early while keeping its drum groove
        # locked under Track B's incoming melody, then swap bass & drums cleanly
        a_harm = np.vstack([librosa.effects.harmonic(ch) for ch in ov_a])
        a_perc = ov_a - a_harm
        b_harm = np.vstack([librosa.effects.harmonic(ch) for ch in ov_b])
        b_perc = ov_b - b_harm
        # Strip A's harmonic layer in the first 45%, bring B's harmonic in over 15%..75%
        g_a_harm = np.cos(0.5 * np.pi * _smoothstep(t / 0.45))
        g_b_harm = np.sin(0.5 * np.pi * _smoothstep((t - 0.10) / 0.60))
        # Swap percussion grooves around the bass-swap point
        g_a_perc = np.cos(0.5 * np.pi * _smoothstep((t - 0.30) / 0.55))
        g_b_perc = np.sin(0.5 * np.pi * _smoothstep((t - 0.15) / 0.55))
        # Re-enforce clean low-band bass-swap so kicks never double-sum
        raw_blend = a_harm * g_a_harm + b_harm * g_b_harm + a_perc * g_a_perc + b_perc * g_b_perc
        _, mid_m, high_m = split_three_band(raw_blend, sr)
        ov_mix = (low_a * gain_a_low + low_b * gain_b_low + mid_m + high_m).astype(np.float32)

    elif spec.transition_style == "long_blend":
        # DAFx-2022 club 3-band EQ weave: highs enter early -> mids cross -> bass swap -> highs out
        g_b_high = np.sin(0.5 * np.pi * _smoothstep(t / 0.45))
        g_a_high = np.cos(0.5 * np.pi * _smoothstep((t - 0.40) / 0.60))
        g_b_mid = np.sin(0.5 * np.pi * _smoothstep((t - 0.15) / 0.65))
        g_a_mid = np.cos(0.5 * np.pi * _smoothstep((t - 0.20) / 0.65))
        ov_mix = (
            low_a * gain_a_low
            + low_b * gain_b_low
            + mid_a * g_a_mid
            + mid_b * g_b_mid
            + high_a * g_a_high
            + high_b * g_b_high
        ).astype(np.float32)

    else:
        # bass_swap / standard_blend: DJ 3-band EQ with mid-carve + smooth HPF exit on Track A
        g_b_high = np.sin(0.5 * np.pi * _smoothstep(t / 0.35))
        g_b_mid = np.sin(0.5 * np.pi * _smoothstep((t - 0.10) / 0.55))
        g_a_mid = np.cos(0.5 * np.pi * _smoothstep((t - 0.25) / 0.60))
        g_a_high = np.cos(0.5 * np.pi * _smoothstep((t - 0.45) / 0.55))
        ov_mix = (
            low_a * gain_a_low
            + low_b * gain_b_low
            + mid_a * g_a_mid
            + mid_b * g_b_mid
            + high_a * g_a_high
            + high_b * g_b_high
        ).astype(np.float32)

    mix_2d = np.concatenate([seg_a_full[:, :pre_n], ov_mix, seg_b_full[:, ov_n:]], axis=1)
    if spec.enable_limiter:
        mix_2d = _soft_limit(mix_2d, ceiling=0.94)

    mix_out = mix_2d if is_stereo_a else mix_2d[0]
    if out_wav is not None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        wav_data = mix_2d.T if is_stereo_a else mix_2d[0]
        sf.write(str(out_wav), wav_data, sr, subtype="PCM_16")

    return RenderResult(
        mix=mix_out,
        track_a_overlap_stem=_to_mono(ov_a),
        track_b_overlap_stem=_to_mono(ov_b),
        sample_rate=sr,
        overlap_start_sample=pre_n,
        overlap_end_sample=pre_n + ov_n,
        tempo_stretch_pct=tempo_stretch_pct,
        phase_correction_ms=round(phase_corr_samples * 1000.0 / float(sr), 2),
        out_wav=out_wav,
    )
