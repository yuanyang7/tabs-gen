"""Stage 1: Source separation via Demucs.

Separates a mixed audio file into individual stems:
  vocals, guitar, bass, drums, piano, other

Uses the htdemucs_6s model (6-stem variant) which explicitly isolates
guitar and piano from the generic "other" bucket.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Stems produced by htdemucs_6s
STEMS_6S = ("vocals", "drums", "bass", "guitar", "piano", "other")
# Stems produced by htdemucs (4-stem fallback)
STEMS_4S = ("vocals", "drums", "bass", "other")

DEFAULT_MODEL = "htdemucs_6s"


def separate(
    audio_path: str | Path,
    output_dir: str | Path,
    model: str = DEFAULT_MODEL,
    device: str = "mps",
    shifts: int = 1,
) -> dict[str, Path]:
    """Run Demucs source separation on an audio file.

    Args:
        audio_path: path to input audio (MP3, WAV, FLAC, …).
        output_dir: directory where per-stem WAV files are written.
        model: Demucs model name. htdemucs_6s recommended for 6 stems.
        device: torch device string. Use "mps" for Apple Silicon,
                "cuda" for NVIDIA GPU, "cpu" as fallback.
        shifts: number of random shifts for test-time augmentation.
                1 = no TTA (fast), 10 = best quality (slow).

    Returns:
        Dict mapping stem name → Path to the separated WAV file.
    """
    import shutil
    import subprocess
    import sys

    if shutil.which("demucs") is None and not Path(sys.executable).parent.joinpath("demucs").exists():
        try:
            import demucs  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "demucs is required for source separation. "
                "Install it with: pip install demucs"
            ) from e

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the demucs executable (handles venv installs)
    demucs_exe = Path(sys.executable).parent / "demucs"
    if not demucs_exe.exists():
        demucs_exe = "demucs"

    cmd = [
        str(demucs_exe),
        "--name", model,
        "--device", device,
        "--shifts", str(shifts),
        "--out", str(output_dir),
        "--two-stems", "no",  # separate all stems
        str(audio_path),
    ]
    # Remove unsupported --two-stems flag; use plain invocation
    cmd = [
        str(demucs_exe),
        "-n", model,
        "-d", device,
        "--shifts", str(shifts),
        "--out", str(output_dir),
        str(audio_path),
    ]

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"demucs failed with exit code {result.returncode}")

    # Demucs writes to: output_dir/<model>/<track_name>/<stem>.wav
    track_name = audio_path.stem
    # Find the actual output directory (demucs may sanitize the track name)
    model_dir = output_dir / model
    if not model_dir.exists():
        raise RuntimeError(f"Expected demucs output dir not found: {model_dir}")

    # Find the stem folder (demucs sanitizes filenames)
    stem_dirs = list(model_dir.iterdir())
    if not stem_dirs:
        raise RuntimeError(f"No output found in {model_dir}")
    # Pick the directory most recently written by this demucs run
    stem_dir = max(stem_dirs, key=lambda d: d.stat().st_mtime)

    stem_paths: dict[str, Path] = {}
    for wav_file in stem_dir.glob("*.wav"):
        stem_name = wav_file.stem  # e.g. "guitar", "bass", "vocals", "drums"
        stem_paths[stem_name] = wav_file
        logger.info("  Found stem %s → %s", stem_name, wav_file)

    return stem_paths


def separate_to_arrays(
    audio_path: str | Path,
    model: str = DEFAULT_MODEL,
    device: str = "mps",
) -> dict[str, tuple[np.ndarray, int]]:
    """Separate audio and return stems as numpy arrays instead of files.

    Returns:
        Dict mapping stem name → (waveform_array, sample_rate).
        waveform_array has shape (channels, n_samples), float32.
    """
    import tempfile

    audio_path = Path(audio_path)
    with tempfile.TemporaryDirectory() as tmp:
        stems = separate(audio_path, output_dir=tmp, model=model, device=device)
        result_arrays: dict[str, tuple[np.ndarray, int]] = {}
        import soundfile as sf
        for name, path in stems.items():
            audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
            result_arrays[name] = (audio.T, sr)  # shape: (channels, samples)
    return result_arrays
