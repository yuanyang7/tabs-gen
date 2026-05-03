"""Stage 1: Source separation.

Two backends are available:

  demucs (default)
    Runs Demucs htdemucs_6s (6-stem) or htdemucs (4-stem) in a single pass.

  mdx
    Two-pass hybrid for higher-quality 4-stem output:
      Pass 1 — MDX-Net vocal model (Kim_Vocal_2) → vocals + no_vocals
      Pass 2 — Demucs htdemucs on no_vocals → drums + bass + other
    Vocals benefit from the more focused MDX-Net model; drums/bass/other are
    cleaner because vocals are already removed before Demucs runs.
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

DEFAULT_MODEL = "htdemucs"
MDX_VOCALS_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
MDX_INSTRUMENTAL_DEMUCS_MODEL = "htdemucs_ft"


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


def separate_mdx(
    audio_path: str | Path,
    output_dir: str | Path,
    device: str = "mps",
    shifts: int = 1,
) -> dict[str, Path]:
    """4-stem separation using MDX-Net (vocals) + Demucs (instrumental).

    Returns stems: vocals, drums, bass, other.
    """
    import shutil

    try:
        from audio_separator.separator import Separator
    except ImportError as e:
        raise ImportError(
            "audio-separator is required for the MDX backend. "
            "Install it with: pip install 'audio-separator[cpu]'"
        ) from e

    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_mdx_tmp"
    tmp_dir.mkdir(exist_ok=True)

    # Pass 1: MDX-Net vocals model → vocals + no_vocals
    logger.info("MDX pass 1: separating vocals with %s", MDX_VOCALS_MODEL)
    sep = Separator(output_dir=str(tmp_dir), output_format="WAV")
    sep.load_model(MDX_VOCALS_MODEL)
    pass1_files = sep.separate(str(audio_path))

    # audio-separator may return bare filenames; resolve against tmp_dir
    def _resolve(name: str) -> Path:
        p = Path(name)
        return p if p.is_absolute() else tmp_dir / p.name

    # audio-separator names files like:
    #   {track}_(Vocals)_Kim_Vocal_2.wav
    #   {track}_(Instrumental)_Kim_Vocal_2.wav
    vocals_file = next(
        (_resolve(f) for f in pass1_files if "Vocals" in Path(f).name), None
    )
    no_vocals_file = next(
        (_resolve(f) for f in pass1_files if "Instrumental" in Path(f).name), None
    )
    if vocals_file is None or no_vocals_file is None:
        raise RuntimeError(
            f"MDX pass 1 did not produce expected Vocals/Instrumental files. "
            f"Got: {pass1_files}"
        )

    # Pass 2: Demucs htdemucs on no_vocals → drums + bass + other (+ vocals we discard)
    logger.info("MDX pass 2: splitting instrumental into drums/bass/other via Demucs")
    demucs_out = tmp_dir / "demucs"
    instrumental_stems = separate(
        audio_path=no_vocals_file,
        output_dir=demucs_out,
        model=MDX_INSTRUMENTAL_DEMUCS_MODEL,
        device=device,
        shifts=shifts,
    )

    # Assemble final stems
    stem_paths: dict[str, Path] = {}

    vocals_dest = output_dir / "vocals.wav"
    shutil.copy(vocals_file, vocals_dest)
    stem_paths["vocals"] = vocals_dest
    logger.info("  stem vocals → %s", vocals_dest)

    for stem in ("drums", "bass", "other"):
        if stem in instrumental_stems:
            dest = output_dir / f"{stem}.wav"
            shutil.copy(instrumental_stems[stem], dest)
            stem_paths[stem] = dest
            logger.info("  stem %s → %s", stem, dest)

    shutil.rmtree(tmp_dir, ignore_errors=True)
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
