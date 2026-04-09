"""YouTube audio downloader using yt-dlp."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def is_youtube_url(value: str) -> bool:
    """Return True if value looks like a YouTube URL."""
    return value.startswith(("http://", "https://")) and any(
        host in value for host in ("youtube.com", "youtu.be")
    )


def download_audio(url: str, output_dir: Path) -> Path:
    """Download audio from a YouTube URL as MP3 using yt-dlp.

    Args:
        url: YouTube video URL.
        output_dir: Directory to save the downloaded MP3.

    Returns:
        Path to the downloaded MP3 file.

    Raises:
        RuntimeError: If yt-dlp is not installed or the download fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # yt-dlp output template — %(title)s gives us the video title as filename
    output_template = str(output_dir / "%(title)s.%(ext)s")

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", "0",          # best quality
        "--output", output_template,
        "--no-playlist",                 # single video only
        "--print", "after_move:filepath",  # print final path to stdout
        # JS challenge solving — required for YouTube as of 2024
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        url,
    ]

    logger.info("Downloading audio from: %s", url)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        raise RuntimeError(
            "yt-dlp is not installed. Install it with: pip install yt-dlp"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"yt-dlp failed (exit {e.returncode}):\n{e.stderr.strip()}"
        )

    # The last non-empty line is the filepath printed by --print after_move:filepath
    lines = [l.strip() for l in proc.stdout.splitlines() if l.strip()]
    if not lines:
        # Fallback: find the most recently modified mp3 in output_dir
        mp3s = sorted(output_dir.glob("*.mp3"), key=lambda p: p.stat().st_mtime)
        if not mp3s:
            raise RuntimeError("yt-dlp finished but no MP3 file was found.")
        return mp3s[-1]

    return Path(lines[-1])
