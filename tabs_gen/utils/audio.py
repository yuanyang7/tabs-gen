"""Audio I/O helpers: loading, resampling, and writing WAV files."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load an audio file as a float32 mono or stereo numpy array.

    Supports WAV/FLAC natively via soundfile; MP3/AAC/OGG are decoded via
    ffmpeg to a temporary WAV first.

    Returns:
        (samples, sample_rate) where samples is shape (channels, n_samples)
        for stereo or (n_samples,) for mono.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".wav", ".flac", ".aiff", ".aif"):
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.T  # (n_samples, channels) → (channels, n_samples)
    else:
        # Decode via ffmpeg to raw float32 PCM
        cmd = [
            "ffmpeg", "-i", str(path),
            "-f", "f32le", "-ar", str(sr), "-ac", "2", "-",
        ]
        result = subprocess.run(cmd, capture_output=True, check=True)
        raw = np.frombuffer(result.stdout, dtype="float32")
        data = raw.reshape(-1, 2).T  # (2, n_samples)
        file_sr = sr

    if file_sr != sr:
        data = _resample(data, file_sr, sr)

    return data, sr


def write_audio(path: str | Path, data: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    """Write a numpy array to a WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        data = data.T  # (channels, n_samples) → (n_samples, channels)
    sf.write(str(path), data, sr, subtype="FLOAT")


def _resample(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import librosa

    if data.ndim == 1:
        return librosa.resample(data, orig_sr=orig_sr, target_sr=target_sr)
    return np.stack([
        librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr)
        for ch in data
    ])
