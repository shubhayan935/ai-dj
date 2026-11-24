"""Deterministic MIR audio analyzer producing AudioTrackCard metadata."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

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


SectionLabel = Literal["intro", "buildup", "drop", "breakdown", "groove", "outro"]


class TrackSection(BaseModel):
    """One phrase-aligned structural section of a full song."""

    index: int
    label: SectionLabel
    start_beat: int
    end_beat: int
    start_sec: float
    end_sec: float
    energy: float
    bass_density: float
    vocal_ratio: float
    percussion_ratio: float


class AudioTrackCard(BaseModel):
    """MIR-extracted structural and perceptual metadata for an audio track."""

    track_id: str
    audio_path: str
    sample_rate: int
    duration_sec: float
    bpm: float
    bpm_stability: float
    local_bpm_stability: float = 1.0
    beat_times: list[float] = Field(default_factory=list)
    phrase_boundaries_sec: list[float] = Field(default_factory=list)
    key: str
    camelot: str
    key_confidence: float
    integrated_lufs: float
    peak_dbfs: float
    energy_curve: list[float] = Field(default_factory=list)
    vocal_activity_mask: list[bool] = Field(default_factory=list)
    recommended_intro_beat: int | None = None
    recommended_outro_beat: int | None = None
    genres: list[str] = Field(default_factory=list)
    timbre_embedding: list[float] = Field(default_factory=list)
    mean_energy: float = 0.0
    vocal_ratio: float = 0.0
    sections: list[TrackSection] = Field(default_factory=list)


def compute_timbre_embedding(y: np.ndarray, sr: int) -> list[float]:
    """Compute a compact L2-normalized 48-dim sonic/timbre fingerprint (MFCC + spectral stats)."""
    if len(y) < sr:
        return [0.0] * 48
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=4), axis=1)
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))) / (sr / 2.0)
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))) / (sr / 2.0)
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    vec = np.concatenate(
        [mfcc_mean, mfcc_std, contrast, np.array([centroid, rolloff, flatness], dtype=np.float64)]
    ).astype(np.float64)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    return [round(float(x), 5) for x in vec]


def timbre_cosine_similarity(a: AudioTrackCard, b: AudioTrackCard) -> float:
    """Cosine similarity in [-1, 1] between two tracks' timbre embeddings."""
    if not a.timbre_embedding or not b.timbre_embedding:
        return 0.0
    va = np.asarray(a.timbre_embedding, dtype=np.float64)
    vb = np.asarray(b.timbre_embedding, dtype=np.float64)
    if va.shape != vb.shape:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-8:
        return 0.0
    return round(float(np.dot(va, vb) / denom), 4)


def _compute_local_bpm_stability(beat_times: list[float], window_beats: int = 16) -> float:
    """Fraction of sliding beat windows whose local median IBI stays within ±6% of global median."""
    if len(beat_times) < window_beats + 2:
        return 0.0
    ibis = np.diff(np.asarray(beat_times, dtype=np.float64))
    global_med = float(np.median(ibis))
    if global_med <= 1e-6:
        return 0.0
    stable = 0
    total = 0
    step = max(4, window_beats // 2)
    for start in range(0, len(ibis) - window_beats + 1, step):
        loc_med = float(np.median(ibis[start : start + window_beats]))
        if abs(loc_med - global_med) / global_med <= 0.06:
            stable += 1
        total += 1
    return round(stable / max(1, total), 4)


def camelot_distance(c1: str, c2: str) -> int:
    """Compute harmonic distance on the Camelot wheel (0=exact, 1=adjacent/relative)."""
    n1, l1 = int(c1[:-1]), c1[-1].upper()
    n2, l2 = int(c2[:-1]), c2[-1].upper()
    ring_dist = min((n1 - n2) % 12, (n2 - n1) % 12)
    return ring_dist + (0 if l1 == l2 else 1)


def load_audio_stereo(audio_path: Path, sr: int = 44100) -> np.ndarray:
    """Decode any audio file (WAV/MP3/FLAC) to float32 stereo (2, N) at `sr` via ffmpeg."""
    try:
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(audio_path),
            "-f",
            "f32le",
            "-ac",
            "2",
            "-ar",
            str(sr),
            "pipe:1",
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True)
        raw = np.frombuffer(proc.stdout, dtype=np.float32)
        if len(raw) >= 2:
            n_frames = len(raw) // 2
            return raw[: n_frames * 2].reshape(n_frames, 2).T.copy()
    except Exception:
        pass
    y, _ = librosa.load(str(audio_path), sr=sr, mono=False)
    if y.ndim == 1:
        return np.vstack([y, y]).astype(np.float32)
    return y[:2].astype(np.float32)


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


def _find_best_cue_beat(
    beat_times: list[float],
    vocal_mask: list[bool],
    energy_curve: list[float],
    *,
    is_intro: bool,
    window_beats: int = 32,
) -> int | None:
    """Pick a phrase-aligned (multiple of 8) cue beat with low vocal clash and steady energy."""
    n_beats = len(beat_times)
    if n_beats < window_beats + 16:
        if is_intro and n_beats > 16:
            return 8
        return max(0, (n_beats // 16) * 8) if n_beats else None
    if is_intro:
        candidates = list(range(8, min(n_beats - window_beats - 8, int(n_beats * 0.35)), 8))
    else:
        max_outro = ((n_beats - 1 - window_beats) // 8) * 8
        lo = min(max_outro, max(32, (int(n_beats * 0.55) // 8) * 8))
        candidates = list(range(lo, max_outro + 1, 8)) if max_outro >= 8 else []
    if not candidates:
        if is_intro:
            return 8
        fallback = ((n_beats - 1 - window_beats) // 8) * 8
        return max(0, fallback)

    best_beat = candidates[0]
    best_score = -1e9
    for b in candidates:
        t0 = int(beat_times[b])
        t1 = int(beat_times[min(n_beats - 1, b + window_beats)])
        if t1 <= t0:
            continue
        v_slice = vocal_mask[t0 : min(len(vocal_mask), t1)]
        e_slice = energy_curve[t0 : min(len(energy_curve), t1)]
        vocal_frac = sum(v_slice) / max(1, len(v_slice))
        mean_e = sum(e_slice) / max(1, len(e_slice))
        # Prefer low vocal activity and solid groove energy (avoid dead silence < 0.25)
        score = (1.0 - vocal_frac) * 2.0 + (1.0 if mean_e >= 0.30 else -1.5) - abs(mean_e - 0.65)
        if score > best_score:
            best_score = score
            best_beat = b
    return best_beat


def segment_track_sections(
    y: np.ndarray,
    sr: int,
    beat_times: list[float],
    energy_curve: list[float],
    vocal_mask: list[bool],
    onset_env: np.ndarray | None = None,
    onset_hop_length: int = 512,
) -> list[TrackSection]:
    """Segment a full track into phrase-aligned structural sections with musical features."""
    n_beats = len(beat_times)
    if n_beats < 32 or len(y) < sr * 4:
        return []

    # Low-band (<200 Hz) signal for per-section bass density & phrase-phase alignment
    sos_low = butter(4, min(200.0, sr * 0.45) / (sr / 2.0), btype="low", output="sos")
    y_low = sosfiltfilt(sos_low, y.astype(np.float64))

    # Find the song's true bar-1 / 8-bar phrase phase offset using beat-sync energy novelty
    # instead of slicing blindly at beat 0, 32, 64...
    phrase_beats = 64 if n_beats >= 512 else 32
    beat_rms = np.zeros(n_beats, dtype=np.float64)
    beat_low = np.zeros(n_beats, dtype=np.float64)
    for bi in range(n_beats - 1):
        s0 = int(beat_times[bi] * sr)
        s1 = min(len(y), int(beat_times[bi + 1] * sr))
        if s1 > s0 + 16:
            beat_rms[bi] = float(np.sqrt(np.mean(y[s0:s1].astype(np.float64) ** 2)))
            beat_low[bi] = float(np.sqrt(np.mean(y_low[s0:s1] ** 2)))
    win = 16
    novelty = np.zeros(n_beats, dtype=np.float64)
    for bi in range(win, n_beats - win):
        d_rms = abs(float(np.mean(beat_rms[bi : bi + win]) - np.mean(beat_rms[bi - win : bi])))
        d_low = abs(float(np.mean(beat_low[bi : bi + win]) - np.mean(beat_low[bi - win : bi])))
        novelty[bi] = d_rms + 1.5 * d_low

    best_phase = 0
    best_phase_score = -1.0
    for phase in range(min(32, max(1, n_beats // 4))):
        pts = list(range(phase, n_beats, 32))
        if len(pts) < 2:
            continue
        score = float(np.sum(novelty[pts]))
        if score > best_phase_score:
            best_phase_score = score
            best_phase = phase

    boundaries: list[int] = [0]
    first_b = best_phase if best_phase >= 16 else (best_phase + 32)
    cur = first_b
    while cur < n_beats - 16:
        # Snap to strongest novelty within ±1 beat of the 32/64-beat phrase grid
        step = phrase_beats
        if phrase_beats == 64 and cur + 32 < n_beats - 16:
            # Allow a 32-beat phrase split when there is a major structural change halfway
            mid_nov = float(novelty[cur + 32])
            if mid_nov > float(np.percentile(novelty[win : n_beats - win], 82)):
                step = 32
        boundaries.append(cur)
        cur += step
    if len(boundaries) < 2:
        boundaries = list(range(0, n_beats, phrase_beats))
    if len(boundaries) < 2:
        return []
    # Merge short (<16 beat) trailing tails into the preceding final section
    if (n_beats - 1) - boundaries[-1] < 16:
        boundaries.pop()

    if onset_env is None:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=onset_hop_length)
    onset_times = librosa.frames_to_time(
        np.arange(len(onset_env)), sr=sr, hop_length=onset_hop_length
    )
    onset_p90 = float(np.percentile(onset_env, 90)) if len(onset_env) else 1.0
    onset_scale = max(onset_p90, 1e-6)

    raw_secs: list[dict] = []
    for i, b0 in enumerate(boundaries):
        b1 = boundaries[i + 1] if i + 1 < len(boundaries) else n_beats - 1
        if b1 - b0 < 16:
            continue
        t0 = float(beat_times[b0])
        t1 = float(beat_times[b1])
        s0, s1 = int(t0 * sr), min(len(y), int(t1 * sr))
        if s1 <= s0 + 256:
            continue
        seg = y[s0:s1].astype(np.float64)
        seg_low = y_low[s0:s1]
        rms_all = float(np.sqrt(np.mean(seg**2))) + 1e-9
        rms_low = float(np.sqrt(np.mean(seg_low**2)))
        bass_density = round(min(1.0, rms_low / rms_all), 3)

        sec_i0, sec_i1 = int(t0), max(int(t0) + 1, min(len(energy_curve), int(t1)))
        e_slice = energy_curve[sec_i0:sec_i1]
        energy = round(float(np.mean(e_slice)), 3) if e_slice else 0.0
        v_slice = vocal_mask[sec_i0 : min(len(vocal_mask), sec_i1)]
        v_ratio = round(float(np.mean(v_slice)), 3) if v_slice else 0.0

        o_mask = (onset_times >= t0) & (onset_times < t1)
        perc = (
            round(min(1.0, float(np.mean(onset_env[o_mask])) / onset_scale), 3)
            if np.any(o_mask)
            else 0.0
        )
        raw_secs.append(
            {
                "start_beat": int(b0),
                "end_beat": int(b1),
                "start_sec": round(t0, 2),
                "end_sec": round(t1, 2),
                "energy": energy,
                "bass_density": bass_density,
                "vocal_ratio": v_ratio,
                "percussion_ratio": perc,
            }
        )

    if not raw_secs:
        return []

    energies = [s["energy"] for s in raw_secs]
    med_e = float(np.median(energies))
    hi_e = max(0.68, float(np.percentile(energies, 70)))
    lo_e = min(0.50, med_e * 0.80)
    n_s = len(raw_secs)

    sections: list[TrackSection] = []
    for idx, s in enumerate(raw_secs):
        e = s["energy"]
        next_e = raw_secs[idx + 1]["energy"] if idx + 1 < n_s else e
        if idx == 0 or (idx == 1 and e < med_e * 0.90):
            label: SectionLabel = "intro"
        elif idx == n_s - 1 or (idx == n_s - 2 and e < med_e * 0.90):
            label = "outro"
        elif next_e - e >= 0.12 and next_e >= hi_e * 0.92:
            label = "buildup"
        elif e >= hi_e and s["bass_density"] >= 0.30:
            label = "drop"
        elif e <= lo_e:
            label = "breakdown"
        else:
            label = "groove"
        sections.append(TrackSection(index=idx, label=label, **s))
    return sections


def analyze_track(
    audio_path: Path,
    *,
    track_id: str | None = None,
    sr: int = 22050,
) -> AudioTrackCard:
    """Run full deterministic MIR analysis on an audio file."""
    y = load_audio_mono(audio_path, sr=sr)
    duration_sec = round(len(y) / float(sr), 3)

    hop_length = 256
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        start_bpm=122.0,
        tightness=120.0,
    )
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = [
        round(float(t), 4)
        for t in librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    ]

    if len(beat_times) >= 4:
        bt_arr = np.array(beat_times, dtype=np.float64)
        ibis = np.diff(bt_arr)
        med_ibi = float(np.median(ibis))
        bpm_stability = round(
            float(np.mean(np.abs(ibis - med_ibi) / max(med_ibi, 1e-6) < 0.08)), 4
        )
        # Multi-beat span median eliminates integer-frame IBI quantization (<0.05 BPM accuracy)
        span = min(32, max(1, len(bt_arr) // 3))
        span_ibis = (bt_arr[span:] - bt_arr[:-span]) / float(span)
        valid_spans = span_ibis[np.abs(span_ibis - med_ibi) / max(med_ibi, 1e-6) < 0.06]
        precise_ibi = float(np.median(valid_spans)) if valid_spans.size else med_ibi
        bpm = round(60.0 / precise_ibi, 2) if precise_ibi > 0 else round(bpm, 2)
    else:
        bpm = round(bpm, 2)
        bpm_stability = 0.0

    # Fold extreme octave errors (e.g. >165 BPM double-time -> half) into club DJ range
    if bpm > 165.0:
        bpm = round(bpm / 2.0, 2)
        beat_times = beat_times[::2]

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
    rec_intro = _find_best_cue_beat(beat_times, vocal_mask, energy_curve, is_intro=True)
    rec_outro = _find_best_cue_beat(beat_times, vocal_mask, energy_curve, is_intro=False)
    local_stab = _compute_local_bpm_stability(beat_times)
    timbre_emb = compute_timbre_embedding(y, sr)
    mean_energy = round(float(np.mean(energy_curve)), 4) if energy_curve else 0.0
    vocal_ratio = round(float(np.mean(vocal_mask)), 4) if vocal_mask else 0.0
    sections = segment_track_sections(
        y,
        sr,
        beat_times,
        energy_curve,
        vocal_mask,
        onset_env=onset_env,
        onset_hop_length=hop_length,
    )

    return AudioTrackCard(
        track_id=track_id or audio_path.stem,
        audio_path=str(audio_path),
        sample_rate=sr,
        duration_sec=duration_sec,
        bpm=bpm,
        bpm_stability=bpm_stability,
        local_bpm_stability=local_stab,
        beat_times=beat_times,
        phrase_boundaries_sec=phrase_boundaries,
        key=key_name,
        camelot=camelot,
        key_confidence=key_conf,
        integrated_lufs=lufs,
        peak_dbfs=peak_dbfs,
        energy_curve=energy_curve,
        vocal_activity_mask=vocal_mask,
        recommended_intro_beat=rec_intro,
        recommended_outro_beat=rec_outro,
        timbre_embedding=timbre_emb,
        mean_energy=mean_energy,
        vocal_ratio=vocal_ratio,
        sections=sections,
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
    for item in manifest:
        audio_path = Path(item["audio_path"])
        if not audio_path.is_absolute() and not audio_path.exists():
            candidate = manifest_path.parent / audio_path.name
            if candidate.exists():
                audio_path = candidate
        card = analyze_track(audio_path, track_id=item["track_id"], sr=sr)
        if item.get("genres"):
            card = card.model_copy(update={"genres": list(item["genres"])})
        cards.append(card)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=out_jsonl.parent,
        prefix=f".{out_jsonl.name}.",
        suffix=".tmp",
        delete=False,
    ) as fh:
        tmp_path = Path(fh.name)
        for card in cards:
            fh.write(card.model_dump_json() + "\n")
    tmp_path.replace(out_jsonl)
    return cards
