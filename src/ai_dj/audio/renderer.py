"""Deterministic MixPlan -> WAV DSP renderer with beatmatching and 3-band EQ bass-swap."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pydantic import BaseModel, Field
from scipy.signal import butter, sosfiltfilt

from ai_dj.audio.analyzer import AudioTrackCard, load_audio_mono

RenderStyle = Literal["quick_cut", "standard_blend", "long_blend", "bass_swap"]


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

    mix: np.ndarray
    track_a_overlap_stem: np.ndarray
    track_b_overlap_stem: np.ndarray
    sample_rate: int
    overlap_start_sample: int
    overlap_end_sample: int
    tempo_stretch_pct: float
    out_wav: Path | None = None


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
    low = sosfiltfilt(sos_low, x)
    high = sosfiltfilt(sos_high, x)
    mid = x - low - high
    return low, mid, high


def normalize_lufs(y: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """Loudness-normalize mono audio to `target_lufs` (ITU-R BS.1770) with safe gain bounds."""
    meter = pyln.Meter(sr)
    x = y.astype(np.float64)
    if len(x) >= sr:
        cur = float(meter.integrated_loudness(x))
    else:
        rms = float(np.sqrt(np.mean(x**2)) + 1e-9)
        cur = 20.0 * np.log10(rms)
    if not np.isfinite(cur) or cur < -60.0:
        return y.astype(np.float32)
    gain_db = max(-18.0, min(18.0, target_lufs - cur))
    return (x * (10.0 ** (gain_db / 20.0))).astype(np.float32)


def _soft_limit(y: np.ndarray, ceiling: float = 0.94) -> np.ndarray:
    """Gentle headroom soft-clipper at -0.54 dBFS to prevent digital clipping."""
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    if peak <= ceiling:
        return y
    return (ceiling * np.tanh(y / ceiling)).astype(np.float32)


def _beat_time(card: AudioTrackCard, beat_idx: int) -> float:
    if card.beat_times and 0 <= beat_idx < len(card.beat_times):
        return card.beat_times[beat_idx]
    beat_dur = 60.0 / max(card.bpm, 60.0)
    return beat_idx * beat_dur


def render_transition(
    card_a: AudioTrackCard,
    card_b: AudioTrackCard,
    spec: RenderSpec,
    out_wav: Path | None = None,
    *,
    sr: int = 22050,
    y_a: np.ndarray | None = None,
    y_b: np.ndarray | None = None,
) -> RenderResult:
    """Render a beatmatched, loudness-matched, 3-band EQ transition from Track A to Track B."""
    audio_a = y_a if y_a is not None else load_audio_mono(Path(card_a.audio_path), sr=sr)
    audio_b = y_b if y_b is not None else load_audio_mono(Path(card_b.audio_path), sr=sr)

    beat_dur_a = 60.0 / max(card_a.bpm, 60.0)
    beat_dur_b = 60.0 / max(card_b.bpm, 60.0)

    # Compute tempo-stretch rate to match Track B's BPM to Track A's BPM
    rate = float(card_a.bpm) / max(float(card_b.bpm), 1.0)
    tempo_stretch_pct = round(abs(rate - 1.0) * 100.0, 3)

    t_cue_a = _beat_time(card_a, spec.cue_out_beat_a)
    t_cue_b = _beat_time(card_b, spec.cue_in_beat_b)

    pre_sec = spec.context_beats * beat_dur_a
    ov_sec = spec.overlap_beats * beat_dur_a
    post_sec = spec.context_beats * beat_dur_a

    pre_n = int(round(pre_sec * sr))
    ov_n = max(sr // 2, int(round(ov_sec * sr)))
    post_n = int(round(post_sec * sr))

    # Extract Track A segment: [cue_out - pre, cue_out + overlap]
    a_cue_sample = int(round(t_cue_a * sr))
    a_start = max(0, a_cue_sample - pre_n)
    a_end = a_cue_sample + ov_n
    seg_a_full = np.zeros(pre_n + ov_n, dtype=np.float32)
    src_a = audio_a[a_start : min(len(audio_a), a_end)]
    offset_a = pre_n - (a_cue_sample - a_start)
    seg_a_full[offset_a : offset_a + len(src_a)] = src_a[: len(seg_a_full) - offset_a]
    seg_a_full = normalize_lufs(seg_a_full, sr, spec.target_lufs)

    # Extract Track B segment starting at cue_in_beat_b, time-stretched to Track A's BPM
    b_needed_orig_sec = (ov_sec + post_sec) * rate + 2.0 * beat_dur_b
    b_start = max(0, int(round(t_cue_b * sr)))
    b_end = min(len(audio_b), b_start + int(round(b_needed_orig_sec * sr)))
    raw_b = audio_b[b_start:b_end]
    if len(raw_b) < sr // 2:
        raw_b = np.pad(raw_b, (0, max(0, sr - len(raw_b))))

    if abs(rate - 1.0) > 0.002 and 0.75 <= rate <= 1.33:
        stretched_b = librosa.effects.time_stretch(raw_b.astype(np.float32), rate=rate)
    else:
        stretched_b = raw_b.astype(np.float32)

    seg_b_full = np.zeros(ov_n + post_n, dtype=np.float32)
    shift_samples = int(round((spec.inject_beat_shift_ms / 1000.0) * sr))
    if shift_samples >= 0:
        avail = min(len(stretched_b), len(seg_b_full) - shift_samples)
        if avail > 0:
            seg_b_full[shift_samples : shift_samples + avail] = stretched_b[:avail]
    else:
        src_off = min(len(stretched_b), -shift_samples)
        avail = min(len(stretched_b) - src_off, len(seg_b_full))
        if avail > 0:
            seg_b_full[:avail] = stretched_b[src_off : src_off + avail]
    seg_b_full = normalize_lufs(seg_b_full, sr, spec.target_lufs)

    # Split overlap segments into 3 EQ bands
    ov_a = seg_a_full[pre_n : pre_n + ov_n]
    ov_b = seg_b_full[:ov_n]
    low_a, mid_a, high_a = split_three_band(ov_a, sr)
    low_b, mid_b, high_b = split_three_band(ov_b, sr)

    t = np.linspace(0.0, 1.0, ov_n, dtype=np.float64)
    if spec.transition_style == "quick_cut":
        # Fast 1-beat equal-power cut centered at the start of the overlap
        cut_n = min(ov_n, max(128, int(round(beat_dur_a * sr))))
        gain_b_mh = np.ones(ov_n, dtype=np.float64)
        gain_a_mh = np.zeros(ov_n, dtype=np.float64)
        ramp = np.linspace(0.0, 1.0, cut_n, dtype=np.float64)
        gain_a_mh[:cut_n] = np.cos(0.5 * np.pi * ramp)
        gain_b_mh[:cut_n] = np.sin(0.5 * np.pi * ramp)
        gain_a_low = gain_a_mh.copy()
        gain_b_low = gain_b_mh.copy()
    else:
        # Smooth equal-power crossfade on mids & highs
        gain_a_mh = np.cos(0.5 * np.pi * t)
        gain_b_mh = np.sin(0.5 * np.pi * t)
        if spec.enable_bass_swap:
            # Clean 2-beat low-band swap at bass_swap_beat_frac so kicks never double-sum
            swap_width = max(0.04, min(0.15, (2.0 * beat_dur_a) / max(ov_sec, 1e-3)))
            s = np.clip((t - (spec.bass_swap_beat_frac - swap_width / 2.0)) / swap_width, 0.0, 1.0)
            gain_a_low = np.cos(0.5 * np.pi * s)
            gain_b_low = np.sin(0.5 * np.pi * s)
        else:
            # Unswapped full-level bass overlap (used to test bass-clash defect detection)
            gain_a_low = np.ones(ov_n, dtype=np.float64)
            gain_b_low = np.ones(ov_n, dtype=np.float64)

    ov_mix = (
        low_a * gain_a_low
        + low_b * gain_b_low
        + (mid_a + high_a) * gain_a_mh
        + (mid_b + high_b) * gain_b_mh
    ).astype(np.float32)

    mix = np.concatenate([seg_a_full[:pre_n], ov_mix, seg_b_full[ov_n:]])
    if spec.enable_limiter:
        mix = _soft_limit(mix, ceiling=0.94)

    if out_wav is not None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_wav), mix, sr, subtype="PCM_16")

    return RenderResult(
        mix=mix,
        track_a_overlap_stem=ov_a,
        track_b_overlap_stem=ov_b,
        sample_rate=sr,
        overlap_start_sample=pre_n,
        overlap_end_sample=pre_n + ov_n,
        tempo_stretch_pct=tempo_stretch_pct,
        out_wav=out_wav,
    )
