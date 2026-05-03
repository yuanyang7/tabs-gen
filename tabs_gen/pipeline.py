"""Pipeline orchestrator: runs all 4 stages end-to-end.

Usage:
    from tabs_gen.pipeline import run_pipeline, PipelineConfig

    config = PipelineConfig(audio_path="song.mp3", output_dir="./output")
    result = run_pipeline(config)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    audio_path: str | Path
    output_dir: str | Path = "./output"

    # Stage 1 — separation
    separation_backend: str = "demucs"  # "demucs" | "mdx"
    demucs_model: str = "htdemucs"
    device: str = "mps"
    demucs_shifts: int = 1

    # Stage 2 — transcription
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    crepe_model: str = "medium"
    vocal_confidence_threshold: float = 0.5

    # Stage 4 — output
    formats: list[str] = field(default_factory=lambda: ["ascii", "gp5"])

    # Instruments to generate
    instruments: list[str] = field(
        default_factory=lambda: ["guitar", "bass", "drums", "vocals"]
    )

    # Tab generation is opt-in; source separation runs by default
    generate_tabs: bool = False

    # Stem compression (WAV → MP3) runs automatically after separation
    compress_stems: bool = True
    stem_bitrate: str = "320k"
    keep_wav: bool = False  # if True, keep WAV stems alongside MP3s

    # Song title for output headers
    title: str = ""

    def __post_init__(self) -> None:
        self.audio_path = Path(self.audio_path)
        self.output_dir = Path(self.output_dir)
        if not self.title:
            self.title = self.audio_path.stem


@dataclass
class PipelineResult:
    stem_paths: dict[str, Path] = field(default_factory=dict)
    mp3_stem_paths: dict[str, Path] = field(default_factory=dict)
    ascii_path: Path | None = None
    gp5_path: Path | None = None
    elapsed_seconds: float = 0.0


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute the full music-to-tabs pipeline.

    Args:
        config: pipeline configuration.

    Returns:
        PipelineResult with paths to generated output files.
    """
    start = time.time()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = config.output_dir / "stems"

    result = PipelineResult()

    # ------------------------------------------------------------------ #
    # Stage 1: Source separation
    # ------------------------------------------------------------------ #
    logger.info("=== Stage 1: Source separation (%s) ===", config.separation_backend)
    if config.separation_backend == "mdx":
        from tabs_gen.stages.separation import separate_mdx

        stem_paths = separate_mdx(
            audio_path=config.audio_path,
            output_dir=stems_dir,
            device=config.device,
            shifts=config.demucs_shifts,
        )
    else:
        from tabs_gen.stages.separation import separate

        stem_paths = separate(
            audio_path=config.audio_path,
            output_dir=stems_dir,
            model=config.demucs_model,
            device=config.device,
            shifts=config.demucs_shifts,
        )
    result.stem_paths = stem_paths

    # ------------------------------------------------------------------ #
    # Stage 1b: Compress stems WAV → MP3
    # ------------------------------------------------------------------ #
    if config.compress_stems:
        logger.info("=== Stage 1b: Compressing stems to MP3 (%s) ===", config.stem_bitrate)
        from tabs_gen.utils.compress import compress_stems as do_compress

        result.mp3_stem_paths = do_compress(
            stem_paths,
            bitrate=config.stem_bitrate,
            keep_wav=config.keep_wav,
        )

    if not config.generate_tabs:
        logger.info("Tab generation disabled (pass generate_tabs=True to enable).")
        result.elapsed_seconds = time.time() - start
        return result

    requested = set(config.instruments)

    # ------------------------------------------------------------------ #
    # Stage 2 + 3: Transcription → Notation mapping
    # ------------------------------------------------------------------ #
    logger.info("=== Stage 2+3: Transcription & notation mapping ===")
    guitar_tab = None
    bass_tab = None
    drum_grid = None
    vocal_staff = None
    tempo = 120.0

    if "guitar" in requested and "guitar" in stem_paths:
        from tabs_gen.stages.transcription import transcribe_guitar
        from tabs_gen.stages.notation.guitar import assign_frets as guitar_assign

        logger.info("Transcribing guitar…")
        guitar_midi = transcribe_guitar(
            stem_paths["guitar"],
            onset_threshold=config.onset_threshold,
            frame_threshold=config.frame_threshold,
        )
        tempo = guitar_midi.tempo or tempo
        logger.info("  %d notes detected (tempo %.1f BPM)", len(guitar_midi.notes), tempo)
        guitar_tab = guitar_assign(guitar_midi)

    if "bass" in requested and "bass" in stem_paths:
        from tabs_gen.stages.transcription import transcribe_bass
        from tabs_gen.stages.notation.bass import assign_frets as bass_assign

        logger.info("Transcribing bass…")
        bass_midi = transcribe_bass(
            stem_paths["bass"],
            onset_threshold=config.onset_threshold,
            frame_threshold=config.frame_threshold,
        )
        if not tempo or tempo == 120.0:
            tempo = bass_midi.tempo or tempo
        logger.info("  %d notes detected", len(bass_midi.notes))
        bass_tab = bass_assign(bass_midi)

    if "drums" in requested and "drums" in stem_paths:
        from tabs_gen.stages.transcription import transcribe_drums
        from tabs_gen.stages.notation.drums import build_drum_grid

        logger.info("Transcribing drums…")
        drum_data = transcribe_drums(stem_paths["drums"])
        if not tempo or tempo == 120.0:
            tempo = drum_data.tempo or tempo
        logger.info("  %d drum events detected", len(drum_data.onsets))
        drum_grid = build_drum_grid(drum_data)

    if "vocals" in requested and "vocals" in stem_paths:
        from tabs_gen.stages.transcription import transcribe_vocals
        from tabs_gen.stages.notation.vocals import build_vocal_staff

        logger.info("Transcribing vocals…")
        vocal_midi = transcribe_vocals(
            stem_paths["vocals"],
            model_capacity=config.crepe_model,
            confidence_threshold=config.vocal_confidence_threshold,
        )
        logger.info("  %d vocal notes detected", len(vocal_midi.notes))
        vocal_staff = build_vocal_staff(vocal_midi)

    # ------------------------------------------------------------------ #
    # Stage 4: Output
    # ------------------------------------------------------------------ #
    logger.info("=== Stage 4: Rendering output ===")

    if "ascii" in config.formats:
        from tabs_gen.stages.output.ascii_tab import render_full_tab

        ascii_text = render_full_tab(
            guitar_tab=guitar_tab,
            bass_tab=bass_tab,
            drum_grid=drum_grid,
            vocal_staff=vocal_staff,
            title=config.title,
        )
        ascii_path = config.output_dir / f"{config.title}.txt"
        ascii_path.write_text(ascii_text, encoding="utf-8")
        result.ascii_path = ascii_path
        logger.info("ASCII tab written: %s", ascii_path)

    if "gp5" in config.formats:
        from tabs_gen.stages.output.gp5 import write_gp5

        gp5_path = config.output_dir / f"{config.title}.gp5"
        write_gp5(
            output_path=gp5_path,
            guitar_tab=guitar_tab,
            bass_tab=bass_tab,
            drum_grid=drum_grid,
            title=config.title,
            tempo=tempo,
        )
        result.gp5_path = gp5_path

    result.elapsed_seconds = time.time() - start
    logger.info("Pipeline complete in %.1fs", result.elapsed_seconds)
    return result
