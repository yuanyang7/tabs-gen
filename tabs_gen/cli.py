"""Command-line interface for tabs-gen."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from tabs_gen.pipeline import PipelineConfig, run_pipeline


@click.command()
@click.argument("audio_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    default="./output",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Directory to write output files.",
)
@click.option(
    "--format", "-f", "formats",
    multiple=True,
    default=["ascii", "gp5"],
    show_default=True,
    type=click.Choice(["ascii", "gp5"], case_sensitive=False),
    help="Output format(s). Can be specified multiple times.",
)
@click.option(
    "--instrument", "-i", "instruments",
    multiple=True,
    default=["guitar", "bass", "drums", "vocals"],
    show_default=True,
    type=click.Choice(["guitar", "bass", "drums", "vocals"], case_sensitive=False),
    help="Instruments to include. Can be specified multiple times.",
)
@click.option(
    "--model",
    default="htdemucs_6s",
    show_default=True,
    help="Demucs separation model. Options: htdemucs, htdemucs_6s, htdemucs_ft.",
)
@click.option(
    "--device",
    default="mps",
    show_default=True,
    help="Torch device for Demucs: mps (Apple Silicon), cuda, cpu.",
)
@click.option(
    "--shifts",
    default=1,
    show_default=True,
    type=int,
    help="Demucs test-time shifts. Higher = better quality, slower (try 4 or 10).",
)
@click.option(
    "--onset-threshold",
    default=0.5,
    show_default=True,
    type=float,
    help="basic-pitch onset detection threshold (0–1).",
)
@click.option(
    "--frame-threshold",
    default=0.3,
    show_default=True,
    type=float,
    help="basic-pitch frame activation threshold (0–1).",
)
@click.option(
    "--crepe-model",
    default="medium",
    show_default=True,
    type=click.Choice(["tiny", "small", "medium", "large", "full"], case_sensitive=False),
    help="CREPE model capacity for vocal transcription.",
)
@click.option(
    "--title",
    default="",
    help="Song title used in output file names and headers (defaults to filename stem).",
)
@click.option(
    "--generate-tabs",
    is_flag=True,
    default=False,
    help="Run transcription and tab generation after source separation (opt-in; output quality is draft-level).",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(
    audio_file: Path,
    output: Path,
    formats: tuple[str, ...],
    instruments: tuple[str, ...],
    model: str,
    device: str,
    shifts: int,
    onset_threshold: float,
    frame_threshold: float,
    crepe_model: str,
    title: str,
    generate_tabs: bool,
    verbose: bool,
) -> None:
    """Split an audio file into instrument stems using Demucs.

    By default only source separation is performed, writing per-instrument WAV
    stems to <output>/stems/. Pass --generate-tabs to also run transcription
    and produce ASCII / Guitar Pro 5 tab files (draft-quality output).

    AUDIO_FILE can be an MP3, WAV, FLAC, or any format supported by ffmpeg.

    Example:

    \b
        tabs-gen song.mp3 --output ./stems/
        tabs-gen song.mp3 --generate-tabs --format ascii --format gp5
        tabs-gen song.wav --generate-tabs --instrument guitar --instrument bass
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        audio_path=audio_file,
        output_dir=output,
        demucs_model=model,
        device=device,
        demucs_shifts=shifts,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        crepe_model=crepe_model,
        formats=list(formats),
        instruments=list(instruments),
        title=title or audio_file.stem,
        generate_tabs=generate_tabs,
    )

    click.echo(f"Processing: {audio_file.name}")
    click.echo(f"Device: {device}  |  Model: {model}")
    if generate_tabs:
        click.echo(f"Instruments: {', '.join(config.instruments)}")
        click.echo(f"Output formats: {', '.join(config.formats)}")
    else:
        click.echo("Mode: stem separation only (use --generate-tabs to also produce tabs)")
    click.echo("")

    try:
        result = run_pipeline(config)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    click.echo("")
    click.echo(f"Done in {result.elapsed_seconds:.1f}s")
    if result.ascii_path:
        click.echo(f"  ASCII tab: {result.ascii_path}")
    if result.gp5_path:
        click.echo(f"  GP5 file:  {result.gp5_path}")
    if result.stem_paths:
        click.echo(f"  Stems dir: {config.output_dir / 'stems'}")


if __name__ == "__main__":
    main()
