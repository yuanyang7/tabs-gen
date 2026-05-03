"""Command-line interface for tabs-gen."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from tabs_gen.pipeline import PipelineConfig, run_pipeline


@click.command()
@click.argument("audio_file", type=str)  # str to accept URLs as well as file paths
@click.option(
    "--output", "-o",
    default="/Volumes/home/tabs-gen-output",
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
    "--backend",
    default="demucs",
    show_default=True,
    type=click.Choice(["demucs", "mdx"], case_sensitive=False),
    help=(
        "Separation backend. "
        "'mdx' chains MDX-Net (vocals) + Demucs (drums/bass/other) for higher quality 4-stem output. "
        "'demucs' runs Demucs alone (supports 4- or 6-stem models)."
    ),
)
@click.option(
    "--model",
    default="htdemucs",
    show_default=True,
    help="Demucs model (demucs backend only). Options: htdemucs, htdemucs_ft, htdemucs_6s.",
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
@click.option(
    "--keep-wav",
    is_flag=True,
    default=False,
    help="Keep source WAV stems alongside the compressed MP3s.",
)
@click.option(
    "--upload",
    is_flag=True,
    default=False,
    help="Upload MP3 and stems to Google Drive after processing (requires rclone configured with 'gdrive' remote).",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def main(
    audio_file: str,
    output: Path,
    formats: tuple[str, ...],
    instruments: tuple[str, ...],
    backend: str,
    model: str,
    device: str,
    shifts: int,
    onset_threshold: float,
    frame_threshold: float,
    crepe_model: str,
    title: str,
    generate_tabs: bool,
    keep_wav: bool,
    upload: bool,
    verbose: bool,
) -> None:
    """Split an audio file into instrument stems using Demucs.

    AUDIO_FILE can be a local file path (MP3, WAV, FLAC, …) or a YouTube URL.
    YouTube URLs are downloaded as MP3 via yt-dlp before processing.

    By default only source separation is performed, writing per-instrument WAV
    stems to <output>/stems/. Pass --generate-tabs to also run transcription
    and produce ASCII / Guitar Pro 5 tab files (draft-quality output).

    Example:

    \b
        tabs-gen song.mp3
        tabs-gen https://www.youtube.com/watch?v=XXXXXXXXXXX
        tabs-gen song.mp3 --generate-tabs --format ascii --format gp5
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve the audio source — download if it's a YouTube URL
    from tabs_gen.utils.youtube import is_youtube_url, download_audio

    output_path = Path(output)
    downloaded_mp3: Path | None = None
    if is_youtube_url(audio_file):
        click.echo(f"YouTube URL detected. Downloading audio…")
        try:
            resolved_audio = download_audio(audio_file, output_path)
        except RuntimeError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        downloaded_mp3 = resolved_audio
        click.echo(f"Downloaded: {resolved_audio.name}")
    else:
        resolved_audio = Path(audio_file)
        if not resolved_audio.exists():
            click.echo(f"Error: file not found: {audio_file}", err=True)
            sys.exit(1)

    config = PipelineConfig(
        audio_path=resolved_audio,
        output_dir=output_path,
        separation_backend=backend,
        demucs_model=model,
        device=device,
        demucs_shifts=shifts,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        crepe_model=crepe_model,
        formats=list(formats),
        instruments=list(instruments),
        title=title or resolved_audio.stem,
        generate_tabs=generate_tabs,
        keep_wav=keep_wav,
    )

    click.echo(f"Processing: {resolved_audio.name}")
    if backend == "mdx":
        click.echo(f"Device: {device}  |  Backend: mdx (BSRoformer vocals + htdemucs_ft instrumental)")
    else:
        click.echo(f"Device: {device}  |  Backend: demucs  |  Model: {model}")
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
    if downloaded_mp3:
        click.echo(f"  MP3:       {downloaded_mp3}")
    if result.ascii_path:
        click.echo(f"  ASCII tab: {result.ascii_path}")
    if result.gp5_path:
        click.echo(f"  GP5 file:  {result.gp5_path}")
    if result.stem_paths:
        click.echo(f"  Stems dir: {config.output_dir / 'stems'}")
    if result.mp3_stem_paths:
        click.echo(f"  MP3 stems: {config.output_dir / 'stems'} ({len(result.mp3_stem_paths)} MP3s)")

    if upload:
        from tabs_gen.utils.gdrive import upload as gdrive_upload

        to_upload: list[Path] = []
        if downloaded_mp3:
            to_upload.append(downloaded_mp3)
        stems_dir = config.output_dir / "stems"
        if stems_dir.exists():
            to_upload.append(stems_dir)

        if to_upload:
            click.echo("")
            click.echo("Uploading to Google Drive…")
            try:
                gdrive_upload(to_upload, remote_subfolder=config.title)
                click.echo(f"  Uploaded to Drive: {config.title}/")
            except RuntimeError as e:
                click.echo(f"  Upload error: {e}", err=True)


if __name__ == "__main__":
    main()
