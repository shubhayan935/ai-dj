"""Cached 4-stem source separation (`drums`, `bass`, `vocals`, `other`) via Demucs (`htdemucs`)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

import librosa
import numpy as np

StemName = Literal["drums", "bass", "vocals", "other"]


@lru_cache(maxsize=1)
def _load_demucs_model(model_name: str = "htdemucs"):
    """Load and cache the pretrained Demucs 4-stem model on CPU/CUDA."""
    import torch
    from demucs.pretrained import get_model

    model = get_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, device


def separate_four_stems(
    audio_2d: np.ndarray,
    sr: int,
    *,
    model_name: str = "htdemucs",
) -> dict[StemName, np.ndarray]:
    """Separate a `(channels, samples)` overlap window into 4 stems at `sr`.

    Returns a dict mapping `"drums"`, `"bass"`, `"vocals"`, `"other"` to float32 arrays
    matching the exact input shape `(channels, samples)`.
    """
    import torch
    from demucs.apply import apply_model

    n_ch, orig_len = audio_2d.shape
    model, device = _load_demucs_model(model_name)
    model_sr = int(model.samplerate)

    # Demucs expects 2-channel stereo at model.samplerate (44.1 kHz)
    if n_ch == 1:
        stereo = np.vstack([audio_2d[0], audio_2d[0]]).astype(np.float32)
    else:
        stereo = audio_2d[:2].astype(np.float32)

    if sr != model_sr:
        stereo_rs = np.vstack(
            [librosa.resample(ch, orig_sr=sr, target_sr=model_sr) for ch in stereo]
        ).astype(np.float32)
    else:
        stereo_rs = stereo

    wav_t = torch.from_numpy(stereo_rs).unsqueeze(0).to(device)
    with torch.no_grad():
        sources = apply_model(model, wav_t, shifts=0, split=True, overlap=0.25, progress=False)[0]
    sources_np = sources.cpu().numpy()  # (4, 2, T)

    out: dict[StemName, np.ndarray] = {}
    for idx, name in enumerate(model.sources):
        stem = sources_np[idx]  # (2, T)
        if sr != model_sr:
            stem = np.vstack(
                [librosa.resample(ch, orig_sr=model_sr, target_sr=sr) for ch in stem]
            ).astype(np.float32)
        # Match original length & channel count exactly
        if stem.shape[1] < orig_len:
            stem = np.pad(stem, ((0, 0), (0, orig_len - stem.shape[1])))
        else:
            stem = stem[:, :orig_len]
        if n_ch == 1:
            stem = np.mean(stem, axis=0, keepdims=True).astype(np.float32)
        out[name] = stem.astype(np.float32)  # type: ignore[index]
    return out
