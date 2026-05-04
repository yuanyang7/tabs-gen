#!/usr/bin/env bash
# Setup script for tabs-gen.
# Run once on a new machine: bash setup.sh
# After setup, activate the venv each session: source .venv/bin/activate

set -euo pipefail

PYTHON=${PYTHON:-python3.12}

# ── System dependencies ────────────────────────────────────────────────────────

if command -v brew &>/dev/null; then
    echo "==> Installing system dependencies via Homebrew…"
    brew list ffmpeg &>/dev/null || brew install ffmpeg
    brew list node   &>/dev/null || brew install node    # required by yt-dlp for YouTube
else
    echo "⚠️  Homebrew not found. Please install manually:"
    echo "    ffmpeg  — https://ffmpeg.org/download.html"
    echo "    node    — https://nodejs.org/"
fi

# ── Python virtual environment ─────────────────────────────────────────────────

if ! command -v "$PYTHON" &>/dev/null; then
    echo "❌  $PYTHON not found. Install Python 3.12 and re-run, or set PYTHON=python3.x"
    exit 1
fi

if [ ! -d .venv ]; then
    echo "==> Creating virtual environment with $PYTHON…"
    "$PYTHON" -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip…"
pip install --upgrade pip --quiet

# ── Python packages ────────────────────────────────────────────────────────────

echo "==> Installing tabs-gen + Demucs separation + MDX backend…"
# Core package + the two separation backends (covers the default use case)
pip install -e ".[separation,mdx]"

echo ""
echo "✅  Core setup complete. You can now run:"
echo "    source .venv/bin/activate"
echo "    tabs-gen \"https://youtu.be/<id>\""
echo "    tabs-gen \"https://youtu.be/<id>\" --backend mdx"
echo ""

# ── Optional: tab generation ───────────────────────────────────────────────────

read -r -p "Install tab generation dependencies (transcription + Guitar Pro output)? [y/N] " yn
if [[ "${yn,,}" == "y" ]]; then
    echo "==> Installing transcription dependencies…"
    pip install -e ".[transcription,output]"

    echo "==> Installing drum transcription (madmom/ADTLib — may take a moment)…"
    # madmom needs Cython and numpy before it can build
    pip install "Cython<3" "numpy<2"
    pip install -e ".[drums]"

    echo ""
    echo "✅  Tab generation ready. Add --generate-tabs to any run."
fi

# ── Optional: Google Drive upload ─────────────────────────────────────────────

echo ""
if ! command -v rclone &>/dev/null; then
    echo "ℹ️   For --upload (Google Drive sync), install rclone:"
    if command -v brew &>/dev/null; then
        echo "    brew install rclone && rclone config"
    else
        echo "    https://rclone.org/install/"
    fi
fi
