"""Google Drive upload via rclone."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

REMOTE = "gdrive"


def upload(local_paths: list[Path], remote_subfolder: str) -> None:
    """Upload a list of files/directories to Google Drive via rclone.

    Args:
        local_paths: Files or directories to upload.
        remote_subfolder: Destination subfolder name inside the configured Drive root.

    Raises:
        RuntimeError: If rclone is not installed or the upload fails.
    """
    for path in local_paths:
        dest = f"{REMOTE}:{remote_subfolder}"
        if path.is_dir():
            cmd = ["rclone", "copy", str(path), f"{dest}/{path.name}"]
        else:
            cmd = ["rclone", "copy", str(path), dest]

        logger.info("Uploading %s → Drive:%s/", path.name, remote_subfolder)
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError("rclone is not installed. Install it with: brew install rclone")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"rclone upload failed:\n{e.stderr.strip()}")

    logger.info("Upload complete → Drive:%s/", remote_subfolder)
