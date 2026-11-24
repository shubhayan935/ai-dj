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
    # Parametric stem/band automation controls (optional; model can override defaults)
    hat_morph_end_frac: float | None = Field(default=None, ge=0.15, le=0.90)
    clap_morph_start_frac: float | None = Field(default=None, ge=0.05, le=0.80)
    clap_morph_end_frac: float | None = Field(default=None, ge=0.20, le=0.95)
    harmony_strip_a_end_frac: float | None = Field(default=None, ge=0.15, le=0.95)
    harmony_entry_b_start_frac: float | None = Field(default=None, ge=0.05, le=0.85)
    vocal_entry_b_start_frac: float | None = Field(default=None, ge=0.15, le=0.90)
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
    rendered_a_overlap_stem: np.ndarray | None = None
    rendered_b_overlap_stem: np.ndarray | None = None
    rendered_a_drums: np.ndarray | None = None
    rendered_b_drums: np.ndarray | None = None
    rendered_a_harm: np.ndarray | None = None
    rendered_b_harm: np.ndarray | None = None


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
    """Find sample shift in [-half_beat, +half_beat] that locks Track B's <160 Hz kicks to A."""
    hop = 256 if sr <= 24000 else 512

    # Low-pass below 160 Hz only when BOTH decks have low-end kick energy so we never
    # correlate Track A's kicks against Track B's offbeat high-frequency hats.
    def _lowpass_kick(sig: np.ndarray) -> np.ndarray | None:
        if len(sig) < 256:
            return None
        sos = butter(4, min(160.0 / (0.5 * sr), 0.9), btype="lowpass", output="sos")
        low = sosfiltfilt(sos, sig.astype(np.float64)).astype(np.float32)
        return low if float(np.sqrt(np.mean(low**2))) >= 1e-4 else None

    low_a = _lowpass_kick(ov_a_mono)
    low_b = _lowpass_kick(ov_b_mono)
    if low_a is not None and low_b is not None:
        ka, kb = low_a, low_b
    else:
        ka, kb = ov_a_mono, ov_b_mono
    env_a = librosa.onset.onset_strength(y=ka, sr=sr, hop_length=hop)
    env_b = librosa.onset.onset_strength(y=kb, sr=sr, hop_length=hop)
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
    best_score = -1e9
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            corr = float(np.dot(a[lag:], b[: n - lag]))
        else:
            corr = float(np.dot(a[: n + lag], b[-lag:]))
        # Mild proximity prior so near-zero downbeat beats an ambiguous ±half-beat upbeat
        prior = 1.0 - 0.12 * (abs(lag) / float(max_lag))
        score = corr * prior
        if score > best_score:
            best_score = score
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
    use_demucs: bool = False,
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

    rendered_a_ov: np.ndarray | None = None
    rendered_b_ov: np.ndarray | None = None
    rendered_a_drums_ov: np.ndarray | None = None
    rendered_b_drums_ov: np.ndarray | None = None
    rendered_a_harm_ov: np.ndarray | None = None
    rendered_b_harm_ov: np.ndarray | None = None
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
        ov_mix = (low_a * gain_a_low + low_b * gain_b_low + a_mh * g_a_mh + b_mh * g_b_mh).astype(
            np.float32
        )

    elif spec.transition_style == "stem_strip_blend":
        # True 4-stem mixing (Demucs) or fast HPSS single-kit drum+bass swap:
        # Bring in Track B's harmonic/synth layer over Track A's drums+bass,
        # then swap drums+bass cleanly at `s_swap` so only ONE drum kit plays at a time.
        stems_a = None
        stems_b = None
        if use_demucs:
            try:
                from ai_dj.audio.stems import separate_four_stems

                stems_a = separate_four_stems(ov_a, sr)
                stems_b = separate_four_stems(ov_b, sr)
            except ImportError:
                stems_a = None
                stems_b = None

        if stems_a is not None and stems_b is not None:
            from ai_dj.audio.analyzer import camelot_distance

            cam_dist = camelot_distance(card_a.camelot, card_b.camelot)
            # PROGRESSIVE 3-BAND DRUM MORPH (calibrated to pro house transitions):
            # Split both drum stems into Kick (<180 Hz), Claps/Snares (180-4000 Hz),
            # and Hats/Tops (>4000 Hz) so Drum A gradually transforms into Drum B band-by-band
            # instead of overlaying two full kits OR snapping all bands at once:
            #   1) Hats (>4 kHz) morph FIRST across [0 .. swap_frac]
            #   2) Claps/Snares (180-4000 Hz) morph SECOND across [swap_frac-0.20 .. swap_frac+0.12]
            #   3) Kick (<180 Hz) + Sub-bass swap LAST on the exact quantized downbeat (1-beat swap)
            da_low, da_mid, da_hi = split_three_band(
                stems_a["drums"], sr, low_hz=180.0, high_hz=4000.0
            )
            db_low, db_mid, db_hi = split_three_band(
                stems_b["drums"], sr, low_hz=180.0, high_hz=4000.0
            )

            swap_beat = max(
                1,
                min(
                    max(1, spec.overlap_beats - 1),
                    int(round(spec.bass_swap_beat_frac * spec.overlap_beats)),
                ),
            )
            swap_frac = max(0.25, min(0.75, swap_beat / float(max(spec.overlap_beats, 1))))

            # Stage 1: Hi-hats / top groove (>4 kHz) glide from A -> B first
            hat_end = (
                float(spec.hat_morph_end_frac)
                if spec.hat_morph_end_frac is not None
                else max(0.20, swap_frac)
            )
            s_d_hi = _smoothstep(t / max(0.15, hat_end))
            g_a_d_hi = np.cos(0.5 * np.pi * s_d_hi)
            g_b_d_hi = np.sin(0.5 * np.pi * s_d_hi)

            # Stage 2: Claps / snares (180-4000 Hz) morph smoothly across the midpoint
            mid_start = (
                float(spec.clap_morph_start_frac)
                if spec.clap_morph_start_frac is not None
                else max(0.08, swap_frac - 0.22)
            )
            mid_end = (
                float(spec.clap_morph_end_frac)
                if spec.clap_morph_end_frac is not None
                else min(0.92, mid_start + 0.34)
            )
            mid_dur = max(0.15, mid_end - mid_start)
            s_d_mid = _smoothstep((t - mid_start) / mid_dur)
            g_a_d_mid = np.cos(0.5 * np.pi * s_d_mid)
            g_b_d_mid = np.sin(0.5 * np.pi * s_d_mid)

            # Stage 3: Kick (<180 Hz) swaps cleanly over 1 beat right on the quantized downbeat
            kick_w = max(0.025, min(0.08, beat_dur_a / max(ov_sec, 1e-3)))
            s_d_low = _smoothstep((t - (swap_frac - kick_w / 2.0)) / kick_w)
            g_a_d_low = np.cos(0.5 * np.pi * s_d_low)
            g_b_d_low = np.sin(0.5 * np.pi * s_d_low)

            a_d = da_low * g_a_d_low + da_mid * g_a_d_mid + da_hi * g_a_d_hi
            b_d = db_low * g_b_d_low + db_mid * g_b_d_mid + db_hi * g_b_d_hi

            has_explicit_harm = (
                spec.harmony_strip_a_end_frac is not None
                or spec.harmony_entry_b_start_frac is not None
                or spec.vocal_entry_b_start_frac is not None
            )
            if has_explicit_harm:
                # Model-driven parametric harmony & vocal automation
                a_out = (
                    float(spec.harmony_strip_a_end_frac)
                    if spec.harmony_strip_a_end_frac is not None
                    else (
                        max(0.15, swap_frac - 0.08)
                        if cam_dist >= 3
                        else min(0.90, swap_frac + 0.25)
                    )
                )
                b_in = (
                    float(spec.harmony_entry_b_start_frac)
                    if spec.harmony_entry_b_start_frac is not None
                    else (min(0.85, swap_frac + 0.02) if cam_dist >= 3 else 0.12)
                )
                v_in = (
                    float(spec.vocal_entry_b_start_frac)
                    if spec.vocal_entry_b_start_frac is not None
                    else min(0.85, swap_frac + 0.04)
                )
                if cam_dist >= 3 or a_out <= b_in:
                    g_a_bass = np.cos(0.5 * np.pi * _smoothstep(t / max(0.15, a_out)))
                    g_b_bass = np.sin(
                        0.5 * np.pi * _smoothstep((t - swap_frac) / max(0.15, 0.78 - swap_frac))
                    )
                else:
                    g_a_bass = g_a_d_low
                    g_b_bass = g_b_d_low
                voc_a_end = max(0.12, min(a_out, swap_frac - 0.06))
                g_a_voc = np.cos(0.5 * np.pi * _smoothstep(t / voc_a_end))
                g_b_voc = np.sin(
                    0.5 * np.pi * _smoothstep((t - v_in) / max(0.15, 0.92 - v_in))
                )
                g_a_oth = np.cos(0.5 * np.pi * _smoothstep(t / max(0.15, a_out)))
                g_b_oth = np.sin(0.5 * np.pi * _smoothstep((t - b_in) / max(0.15, 0.88 - b_in)))
            elif cam_dist >= 3:
                # Key-aware DRUM-BRIDGE (distant keys like 1A -> 5A):
                # Strip Track A harmony + bass to 0% before Track B melody enters
                a_out = max(0.15, swap_frac - 0.08)
                b_in = min(0.85, swap_frac + 0.02)
                g_a_bass = np.cos(0.5 * np.pi * _smoothstep(t / a_out))
                g_b_bass = np.sin(
                    0.5 * np.pi * _smoothstep((t - swap_frac) / max(0.15, 0.75 - swap_frac))
                )
                g_a_voc = np.cos(0.5 * np.pi * _smoothstep(t / max(0.12, swap_frac - 0.14)))
                g_b_voc = np.sin(
                    0.5
                    * np.pi
                    * _smoothstep((t - (swap_frac + 0.04)) / max(0.15, 0.88 - swap_frac))
                )
                g_a_oth = np.cos(0.5 * np.pi * _smoothstep(t / a_out))
                g_b_oth = np.sin(0.5 * np.pi * _smoothstep((t - b_in) / max(0.15, 0.86 - b_in)))
            else:
                # Harmonic-compatible house weave (e.g. 5A -> 4A/5A):
                # Bass follows the 1-beat downbeat kick swap; A's synths glide down gently
                # as B's chords/synths enter cleanly over the morphed drum groove.
                g_a_bass = g_a_d_low
                g_b_bass = g_b_d_low
                g_a_voc = np.cos(0.5 * np.pi * _smoothstep(t / max(0.20, swap_frac - 0.06)))
                g_b_voc = np.sin(
                    0.5 * np.pi * _smoothstep((t - swap_frac) / max(0.20, 0.88 - swap_frac))
                )
                g_a_oth = np.cos(
                    0.5 * np.pi * _smoothstep((t - 0.10) / max(0.25, swap_frac + 0.15))
                )
                g_b_oth = np.sin(
                    0.5 * np.pi * _smoothstep((t - 0.12) / max(0.25, 0.78 - 0.12))
                )

            a_h = stems_a["other"] * g_a_oth + stems_a["vocals"] * g_a_voc
            b_h = stems_b["other"] * g_b_oth + stems_b["vocals"] * g_b_voc
            rendered_a_ov = (a_d + stems_a["bass"] * g_a_bass + a_h).astype(np.float32)
            rendered_b_ov = (b_d + stems_b["bass"] * g_b_bass + b_h).astype(np.float32)
            rendered_a_drums_ov = a_d.astype(np.float32)
            rendered_b_drums_ov = b_d.astype(np.float32)
            rendered_a_harm_ov = (a_h + stems_a["bass"] * g_a_bass).astype(np.float32)
            rendered_b_harm_ov = (b_h + stems_b["bass"] * g_b_bass).astype(np.float32)

            # Seamless 2-beat boundary blend with raw ov_a / ov_b at edges
            edge_w = max(0.03, min(0.08, (2.0 * beat_dur_a) / max(ov_sec, 1e-3)))
            in_blend = _smoothstep(t / edge_w)
            out_blend = _smoothstep((1.0 - t) / edge_w)
            stem_mix = rendered_a_ov + rendered_b_ov
            ov_mix = (
                ov_a * (1.0 - in_blend)
                + stem_mix * (in_blend * out_blend)
                + ov_b * (1.0 - out_blend)
            ).astype(np.float32)
        else:
            from ai_dj.audio.analyzer import camelot_distance

            cam_dist = camelot_distance(card_a.camelot, card_b.camelot)
            a_harm = np.vstack([librosa.effects.harmonic(ch) for ch in ov_a])
            a_perc = ov_a - a_harm
            b_harm = np.vstack([librosa.effects.harmonic(ch) for ch in ov_b])
            b_perc = ov_b - b_harm
            # Honor parametric harmony strip/entry controls (key-aware defaults in HPSS path)
            default_a_out = 0.42 if cam_dist >= 3 else 0.50
            default_b_in = 0.52 if cam_dist >= 3 else 0.25
            a_out = (
                float(spec.harmony_strip_a_end_frac)
                if spec.harmony_strip_a_end_frac is not None
                else default_a_out
            )
            b_in = (
                float(spec.harmony_entry_b_start_frac)
                if spec.harmony_entry_b_start_frac is not None
                else default_b_in
            )
            g_a_harm = np.cos(0.5 * np.pi * _smoothstep(t / max(0.15, a_out)))
            g_b_harm = np.sin(0.5 * np.pi * _smoothstep((t - b_in) / max(0.15, 0.88 - b_in)))
            # Progressive 3-band percussive morph in fast path (hats glide, kick+claps swap tight)
            pa_low, pa_mid, pa_hi = split_three_band(a_perc, sr, low_hz=180.0, high_hz=4000.0)
            pb_low, pb_mid, pb_hi = split_three_band(b_perc, sr, low_hz=180.0, high_hz=4000.0)
            hat_end = (
                float(spec.hat_morph_end_frac)
                if spec.hat_morph_end_frac is not None
                else max(0.20, spec.bass_swap_beat_frac)
            )
            hat_start = max(0.0, hat_end - 0.30)
            s_hi = _smoothstep((t - hat_start) / max(0.15, hat_end - hat_start))
            a_p = (
                pa_low * np.cos(0.5 * np.pi * s_swap)
                + pa_mid * np.cos(0.5 * np.pi * s_swap)
                + pa_hi * np.cos(0.5 * np.pi * s_hi)
            )
            b_p = (
                pb_low * np.sin(0.5 * np.pi * s_swap)
                + pb_mid * np.sin(0.5 * np.pi * s_swap)
                + pb_hi * np.sin(0.5 * np.pi * s_hi)
            )
            rendered_a_harm_ov = (a_harm * g_a_harm).astype(np.float32)
            rendered_b_harm_ov = (b_harm * g_b_harm).astype(np.float32)
            rendered_a_drums_ov = a_p.astype(np.float32)
            rendered_b_drums_ov = b_p.astype(np.float32)
            rendered_a_ov = (rendered_a_harm_ov + rendered_a_drums_ov).astype(np.float32)
            rendered_b_ov = (rendered_b_harm_ov + rendered_b_drums_ov).astype(np.float32)
            raw_blend = rendered_a_ov + rendered_b_ov
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
        rendered_a_overlap_stem=_to_mono(rendered_a_ov) if rendered_a_ov is not None else None,
        rendered_b_overlap_stem=_to_mono(rendered_b_ov) if rendered_b_ov is not None else None,
        rendered_a_drums=_to_mono(rendered_a_drums_ov) if rendered_a_drums_ov is not None else None,
        rendered_b_drums=_to_mono(rendered_b_drums_ov) if rendered_b_drums_ov is not None else None,
        rendered_a_harm=_to_mono(rendered_a_harm_ov) if rendered_a_harm_ov is not None else None,
        rendered_b_harm=_to_mono(rendered_b_harm_ov) if rendered_b_harm_ov is not None else None,
    )
