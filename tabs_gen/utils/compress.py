"""Stem WAV → MP3 compression using ffmpeg (pydub as fallback)."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_BITRATE = "320k"


def wav_to_mp3(wav_path: Path, bitrate: str = DEFAULT_BITRATE) -> Path:
    """Convert a single WAV file to MP3, returning the MP3 path."""
    mp3_path = wav_path.with_suffix(".mp3")

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(wav_path),
            "-ab", bitrate,
            str(mp3_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed for {wav_path.name}:\n{result.stderr.strip()}")
    else:
        # Fallback: pydub (requires pydub and ffmpeg installed as a library)
        try:
            from pydub import AudioSegment
        except ImportError:
            raise RuntimeError(
                "ffmpeg not found on PATH and pydub is not installed. "
                "Install ffmpeg (brew install ffmpeg) or pydub (pip install pydub)."
            )
        bitrate_int = bitrate.rstrip("kK")
        AudioSegment.from_wav(str(wav_path)).export(
            str(mp3_path), format="mp3", bitrate=f"{bitrate_int}k"
        )

    return mp3_path


def compress_stems(
    stem_paths: dict[str, Path],
    bitrate: str = DEFAULT_BITRATE,
    keep_wav: bool = True,
) -> dict[str, Path]:
    """Convert all stem WAVs to MP3.

    Args:
        stem_paths: mapping of stem name → WAV path (from separation stage).
        bitrate: MP3 bitrate, e.g. "320k".
        keep_wav: if False, delete the source WAV after conversion.

    Returns:
        Mapping of stem name → MP3 path.
    """
    mp3_paths: dict[str, Path] = {}
    for stem_name, wav_path in stem_paths.items():
        logger.info("  Compressing %s.wav → mp3 (%s)", stem_name, bitrate)
        mp3_path = wav_to_mp3(wav_path, bitrate=bitrate)
        mp3_paths[stem_name] = mp3_path
        if not keep_wav:
            wav_path.unlink()
            logger.info("  Removed %s", wav_path.name)

    return mp3_paths
