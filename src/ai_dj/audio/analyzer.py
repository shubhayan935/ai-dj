"""Deterministic MIR audio analyzer producing AudioTrackCard metadata."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
from pydantic import BaseModel, Field
from scipy.signal import butter, sosfiltfilt

# Krumhansl-Schmuckler pitch-class key profiles (C, C#, D, ..., B)
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)
_PITCH_CLASSES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Camelot Wheel lookup: (pitch_class_index, is_major) -> Camelot code
_MAJOR_CAMELOT = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]
_MINOR_CAMELOT = ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]


class AudioTrackCard(BaseModel):
    """MIR-extracted structural and perceptual metadata for an audio track."""

    track_id: str
    audio_path: str
    sample_rate: int
    duration_sec: float
    bpm: float
    bpm_stability: float
    beat_times: list[float] = Field(default_factory=list)
    phrase_boundaries_sec: list[float] = Field(default_factory=list)
    key: str
    camelot: str
    key_confidence: float
    integrated_lufs: float
    peak_dbfs: float
    energy_curve: list[float] = Field(default_factory=list)
    vocal_activity_mask: list[bool] = Field(default_factory=list)


def camelot_distance(c1: str, c2: str) -> int:
    """Compute harmonic distance on the Camelot wheel (0=exact, 1=adjacent/relative)."""
    n1, l1 = int(c1[:-1]), c1[-1].upper()
    n2, l2 = int(c2[:-1]), c2[-1].upper()
    ring_dist = min((n1 - n2) % 12, (n2 - n1) % 12)
    return ring_dist + (0 if l1 == l2 else 1)


def load_audio_mono(audio_path: Path, sr: int = 22050) -> np.ndarray:
    """Decode any audio file (WAV/MP3/FLAC) to float32 mono at `sr` via soundfile or ffmpeg."""
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        return y.astype(np.float32)
    except Exception:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "pipe:1",
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True)
        return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def estimate_key(y: np.ndarray, sr: int) -> tuple[str, str, float]:
    """Estimate musical key & Camelot code using CQT chroma + Krumhansl-Schmuckler profiles."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    if np.linalg.norm(chroma_mean) < 1e-8:
        return "C major", "8B", 0.0

    best_corr = -2.0
    second_corr = -2.0
    best_idx = 0
    best_is_major = True
    for idx in range(12):
        maj_p = np.roll(_MAJOR_PROFILE, idx)
        min_p = np.roll(_MINOR_PROFILE, idx)
        c_maj = float(np.corrcoef(chroma_mean, maj_p)[0, 1])
        c_min = float(np.corrcoef(chroma_mean, min_p)[0, 1])
        for corr, is_maj in ((c_maj, True), (c_min, False)):
            if corr > best_corr:
                second_corr = best_corr
                best_corr = corr
                best_idx = idx
                best_is_major = is_maj
            elif corr > second_corr:
                second_corr = corr

    key_name = f"{_PITCH_CLASSES[best_idx]} {'major' if best_is_major else 'minor'}"
    camelot = _MAJOR_CAMELOT[best_idx] if best_is_major else _MINOR_CAMELOT[best_idx]
    confidence = round(max(0.0, min(1.0, (best_corr - second_corr) * 3.0 + 0.3)), 4)
    return key_name, camelot, confidence


def estimate_vocal_activity_mask(
    y: np.ndarray,
    sr: int,
    hop_sec: float = 1.0,
) -> list[bool]:
    """Fast deterministic vocal-band presence detector (300-3400 Hz harmonic energy ratio)."""
    nyq = 0.5 * sr
    sos_vocal = butter(4, [300.0 / nyq, min(3400.0 / nyq, 0.95)], btype="bandpass", output="sos")
    y_vocal = sosfiltfilt(sos_vocal, y.astype(np.float64))

    frame_len = max(1, int(round(hop_sec * sr)))
    n_frames = max(1, len(y) // frame_len)
    mask: list[bool] = []
    global_rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-9)

    for i in range(n_frames):
        seg = y[i * frame_len : (i + 1) * frame_len].astype(np.float64)
        seg_v = y_vocal[i * frame_len : (i + 1) * frame_len]
        rms_total = float(np.sqrt(np.mean(seg**2)) + 1e-9)
        rms_vocal = float(np.sqrt(np.mean(seg_v**2)) + 1e-9)
        band_ratio = rms_vocal / rms_total
        # Vocal-band dominance + non-silent energy + tonal (lower spectral flatness)
        if rms_total < 0.15 * global_rms:
            mask.append(False)
            continue
        spec = np.abs(np.fft.rfft(seg_v * np.hanning(len(seg_v)))) + 1e-9
        flatness = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
        mask.append(bool(band_ratio > 0.52 and flatness < 0.28))
    return mask


def measure_lufs_and_peak(y: np.ndarray, sr: int) -> tuple[float, float]:
    """Compute ITU-R BS.1770 integrated LUFS and sample peak in dBFS."""
    meter = pyln.Meter(sr)
    if len(y) < sr:
        rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-9)
        lufs = round(20.0 * np.log10(rms), 2)
    else:
        lufs = float(meter.integrated_loudness(y.astype(np.float64)))
        if not np.isfinite(lufs):
            lufs = -60.0
        lufs = round(lufs, 2)
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    peak_dbfs = round(20.0 * np.log10(max(peak, 1e-9)), 2)
    return lufs, peak_dbfs


def analyze_track(
    audio_path: Path,
    *,
    track_id: str | None = None,
    sr: int = 22050,
) -> AudioTrackCard:
    """Run full deterministic MIR analysis on an audio file."""
    y = load_audio_mono(audio_path, sr=sr)
    duration_sec = round(len(y) / float(sr), 3)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = [round(float(t), 4) for t in librosa.frames_to_time(beat_frames, sr=sr)]

    if len(beat_times) >= 4:
        ibis = np.diff(np.array(beat_times, dtype=np.float64))
        med = float(np.median(ibis))
        bpm = round(60.0 / med, 2) if med > 0 else round(bpm, 2)
        bpm_stability = round(float(np.mean(np.abs(ibis - med) / max(med, 1e-6) < 0.08)), 4)
    else:
        bpm = round(bpm, 2)
        bpm_stability = 0.0

    # 8-bar (32-beat) phrase boundaries aligned to the beatgrid
    phrase_boundaries = [beat_times[i] for i in range(0, len(beat_times), 32)]

    key_name, camelot, key_conf = estimate_key(y, sr)
    lufs, peak_dbfs = measure_lufs_and_peak(y, sr)

    # 1-second normalized RMS energy curve
    hop = sr
    n_sec = max(1, len(y) // hop)
    rms_vals = [
        float(np.sqrt(np.mean(y[i * hop : (i + 1) * hop].astype(np.float64) ** 2)))
        for i in range(n_sec)
    ]
    max_rms = max(rms_vals) if rms_vals and max(rms_vals) > 0 else 1.0
    energy_curve = [round(v / max_rms, 4) for v in rms_vals]

    vocal_mask = estimate_vocal_activity_mask(y, sr, hop_sec=1.0)

    return AudioTrackCard(
        track_id=track_id or audio_path.stem,
        audio_path=str(audio_path),
        sample_rate=sr,
        duration_sec=duration_sec,
        bpm=bpm,
        bpm_stability=bpm_stability,
        beat_times=beat_times,
        phrase_boundaries_sec=phrase_boundaries,
        key=key_name,
        camelot=camelot,
        key_confidence=key_conf,
        integrated_lufs=lufs,
        peak_dbfs=peak_dbfs,
        energy_curve=energy_curve,
        vocal_activity_mask=vocal_mask,
    )


def analyze_audio_pool(
    manifest_path: Path,
    out_jsonl: Path,
    *,
    sr: int = 22050,
) -> list[AudioTrackCard]:
    """Analyze all tracks in a CC audio pool manifest and write AudioTrackCards to JSONL."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cards: list[AudioTrackCard] = []
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for item in manifest:
            card = analyze_track(Path(item["audio_path"]), track_id=item["track_id"], sr=sr)
            cards.append(card)
            fh.write(card.model_dump_json() + "\n")
    return cards
