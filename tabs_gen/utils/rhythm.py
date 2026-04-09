"""BPM detection and beat grid utilities."""

from __future__ import annotations

import numpy as np


def detect_tempo(audio: np.ndarray, sr: int) -> float:
    """Estimate BPM using librosa's beat tracker.

    Args:
        audio: mono float32 array of shape (n_samples,).
        sr: sample rate.

    Returns:
        Estimated tempo in BPM.
    """
    import librosa

    mono = _to_mono(audio)
    tempo, _ = librosa.beat.beat_track(y=mono, sr=sr)
    return float(tempo)


def detect_beats(audio: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    """Detect beats and estimate tempo.

    Returns:
        (tempo_bpm, beat_times_seconds)
    """
    import librosa

    mono = _to_mono(audio)
    tempo, beat_frames = librosa.beat.beat_track(y=mono, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    return float(tempo), beat_times


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio
