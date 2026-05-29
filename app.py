"""Gradio web UI for tabs-gen.

Run:
    source .venv/bin/activate
    python app.py
Then open http://localhost:7860
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from pathlib import Path

import gradio as gr

STEM_NAMES = ["vocals", "guitar", "bass", "drums", "piano", "other"]

# --------------------------------------------------------------------------- #
# Logging bridge
# --------------------------------------------------------------------------- #

class _QueueHandler(logging.Handler):
    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put(self.format(record))


# --------------------------------------------------------------------------- #
# Preset definitions
# --------------------------------------------------------------------------- #

PRESETS = {
    "fast": dict(
        backend="demucs",
        model="htdemucs",
        shifts=1,
        generate_tabs=False,
        crepe_model="tiny",
        desc="Stem separation only. No tabs. Fastest path — done in ~2 min.",
    ),
    "balanced": dict(
        backend="demucs",
        model="htdemucs_ft",
        shifts=4,
        generate_tabs=False,
        crepe_model="medium",
        desc="Fine-tuned model, stems only. Better separation quality, ~10 min.",
    ),
    "best": dict(
        backend="mdx",
        model="htdemucs_ft",   # not used by MDX, kept for display
        shifts=10,
        generate_tabs=False,
        crepe_model="full",
        desc="MDX backend + 10 shifts. Best stem separation quality, ~30 min.",
    ),
}


def _apply_preset(key: str):
    p = PRESETS[key]
    show_model = p["backend"] == "demucs"
    return (
        gr.update(value=p["backend"]),
        gr.update(value=p["model"], visible=show_model),
        gr.update(value=p["shifts"]),
        gr.update(value=p["generate_tabs"]),
        gr.update(value=p["crepe_model"]),
        gr.update(value=p["desc"]),
    )


# --------------------------------------------------------------------------- #
# Progress bar HTML helper
# --------------------------------------------------------------------------- #

def _prog_html(val: float, desc: str = "") -> str:
    """Return an HTML progress bar string (0.0 – 1.0)."""
    if val <= 0 and not desc:
        return ""
    pct = min(100, max(0, int(val * 100)))
    color = "#22c55e" if pct >= 100 else "#818cf8"
    return (
        '<div style="margin:8px 0 12px;font-family:sans-serif">'
        f'<div style="background:#374151;border-radius:8px;height:20px;overflow:hidden">'
        f'<div style="background:{color};height:100%;width:{pct}%;'
        'transition:width 0.5s ease;border-radius:8px"></div>'
        '</div>'
        f'<div style="font-size:13px;color:#9ca3af;margin-top:5px;display:flex;justify-content:space-between">'
        f'<span>{desc}</span><strong style="color:#e5e7eb">{pct}%</strong></div>'
        '</div>'
    )


# --------------------------------------------------------------------------- #
# Audio player HTML helper
# --------------------------------------------------------------------------- #

def _audio_html(path: str | None, label: str) -> str:
    """Render a native HTML5 audio player for a stem file.

    Uses preload="metadata" so the browser fetches just the audio headers —
    the duration appears immediately in the seek bar without the user having
    to click play first.  The seek bar shows the full song (no scrolling).
    """
    icon = "🎵"
    header = (
        f'<div style="display:flex;align-items:center;justify-content:space-between;'
        f'margin-bottom:6px">'
        f'<span style="font-weight:600;font-size:14px">{icon} {label}</span>'
    )

    if not path:
        return (
            header
            + '<span style="font-size:11px;color:#6b7280">not generated</span></div>'
            + '<div style="background:#1f2937;border-radius:8px;height:48px;'
            'display:flex;align-items:center;justify-content:center;color:#4b5563;font-size:12px">'
            'No audio</div>'
        )

    url = f"/gradio_api/file={path}"
    return (
        header
        + f'<a href="{url}" download style="font-size:11px;color:#818cf8;text-decoration:none">'
        '⬇ download</a></div>'
        f'<audio controls preload="metadata" '
        f'style="width:100%;height:48px;border-radius:8px;outline:none;display:block">'
        f'<source src="{url}" type="audio/mpeg">'
        f'</audio>'
    )


def _stems_html(stem_paths: dict[str, str] | None) -> list[str]:
    """Return a list of HTML strings for all STEM_NAMES, in order."""
    paths = stem_paths or {}
    return [_audio_html(paths.get(name), name.capitalize()) for name in STEM_NAMES]


# --------------------------------------------------------------------------- #
# Pipeline runner (streaming generator)
# --------------------------------------------------------------------------- #

# Map substrings found in log lines → (progress_fraction, label)
_STAGE_PROGRESS: list[tuple[str, float, str]] = [
    ("Stage 1: Source separation",  0.08, "Separating stems…"),
    ("Stage 1b: Compressing",       0.82, "Compressing to MP3…"),
    ("Stage 2+3: Transcription",    0.88, "Transcribing instruments…"),
    ("Stage 4: Rendering",          0.96, "Rendering output…"),
    ("Pipeline complete",           1.00, "Done!"),
]


def _run(
    audio_file: str | None,
    youtube_url: str,
    output_dir: str,
    backend: str,
    model: str,
    device: str,
    shifts: int,
    instruments: list[str],
    formats: list[str],
    generate_tabs: bool,
    keep_wav: bool,
    onset_threshold: float,
    frame_threshold: float,
    crepe_model: str,
    title: str,
):
    """Generator — yields (logs, prog_html, 6×stem_html, downloads, stems_dir_state)."""

    from tabs_gen.pipeline import PipelineConfig, run_pipeline
    from tabs_gen.utils.youtube import download_audio

    log_q: queue.Queue = queue.Queue()
    result_box: dict = {}

    handler = _QueueHandler(log_q)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    )
    root = logging.getLogger()
    prev_level = root.level
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    for _lib in ("matplotlib", "PIL", "numba", "torch", "torchaudio",
                 "urllib3", "filelock", "fsspec", "audioread", "resampy"):
        logging.getLogger(_lib).setLevel(logging.WARNING)

    def _push_prog(val: float, desc: str = "") -> None:
        log_q.put(("prog", val, desc))

    def _worker() -> None:
        try:
            _title      = (title       or "").strip()
            _output_dir = (output_dir  or "").strip()
            _youtube    = (youtube_url or "").strip()
            out_path = Path(_output_dir or str(Path.home() / "Music" / "tabs-gen"))

            if _youtube:
                log_q.put("[UI] Detected YouTube URL — downloading audio…")
                _push_prog(0.03, "Downloading from YouTube…")
                resolved = download_audio(_youtube, out_path)
                _push_prog(0.07, "Download complete — starting pipeline…")
                song_name = _title or resolved.stem
                song_dir  = out_path / song_name
                song_dir.mkdir(parents=True, exist_ok=True)
                resolved  = resolved.rename(song_dir / resolved.name)

            elif audio_file:
                resolved  = Path(audio_file)
                song_name = _title or resolved.stem
                song_dir  = out_path / song_name

            else:
                result_box["error"] = "Please upload an audio file or paste a YouTube URL."
                return

            cfg = PipelineConfig(
                audio_path=resolved,
                output_dir=song_dir,
                separation_backend=backend,
                demucs_model=model,
                device=device,
                demucs_shifts=int(shifts),
                onset_threshold=float(onset_threshold),
                frame_threshold=float(frame_threshold),
                crepe_model=crepe_model,
                formats=list(formats)     if formats     else ["ascii"],
                instruments=list(instruments) if instruments else list(STEM_NAMES[:4]),
                title=_title or resolved.stem,
                generate_tabs=generate_tabs,
                keep_wav=keep_wav,
            )

            result_box["result"] = run_pipeline(cfg)
            result_box["config"] = cfg

        except Exception as exc:
            import traceback
            result_box["error"] = f"{exc}\n{traceback.format_exc()}"
        finally:
            log_q.put(None)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    log_lines: list[str] = []
    cur_prog = 0.0
    cur_desc = "Starting…"

    _stem_empty = _stems_html(None)   # placeholder HTML for each stem
    _tail_empty: list = [[], None]

    def _partial():
        return ["\n".join(log_lines), _prog_html(cur_prog, cur_desc)] + _stem_empty + _tail_empty

    yield _partial()

    while True:
        try:
            item = log_q.get(timeout=0.25)
        except queue.Empty:
            if not thread.is_alive():
                break
            yield _partial()
            continue

        if item is None:
            break

        if isinstance(item, tuple):
            _, cur_prog, cur_desc = item
            yield _partial()
            continue

        log_lines.append(item)
        for marker, val, desc in _STAGE_PROGRESS:
            if marker in item:
                cur_prog, cur_desc = val, desc
                break
        yield _partial()

    thread.join()
    root.removeHandler(handler)
    root.setLevel(prev_level)

    if "error" in result_box:
        log_lines.append(f"\n❌  {result_box['error']}")
        yield ["\n".join(log_lines), _prog_html(1.0, "Failed ❌")] + _stem_empty + _tail_empty
        return

    result = result_box.get("result")
    cfg    = result_box.get("config")
    if result is None:
        yield _partial()
        return

    mp3_paths: dict[str, Path] = result.mp3_stem_paths or result.stem_paths
    stem_paths_state = {k: str(v) for k, v in mp3_paths.items() if v and v.exists()}

    downloads: list[str] = []
    if result.ascii_path and result.ascii_path.exists():
        downloads.append(str(result.ascii_path))
    if result.gp5_path and result.gp5_path.exists():
        downloads.append(str(result.gp5_path))

    stem_lines = "\n".join(
        f"   {name:<8} → {path}" for name, path in stem_paths_state.items()
    )
    log_lines.append(
        f"\n✅  Done in {result.elapsed_seconds:.1f}s\n"
        f"── Stems saved ──\n{stem_lines}"
        + (f"\n── Tabs ──\n   ASCII → {result.ascii_path}" if result.ascii_path else "")
        + (f"\n   GP5   → {result.gp5_path}"               if result.gp5_path   else "")
    )
    yield (
        ["\n".join(log_lines), _prog_html(1.0, "Done ✅")]
        + _stems_html(stem_paths_state)
        + [downloads, stem_paths_state]
    )


# --------------------------------------------------------------------------- #
# Custom mix
# --------------------------------------------------------------------------- #

def _create_mix(stem_paths_state: dict | None, selected: list[str]) -> tuple[str, str]:
    """Combine selected stems into a single MP3 using ffmpeg amix."""
    if not stem_paths_state:
        return "", "⚠️ Run the pipeline first to generate stems."
    if not selected:
        return "", "⚠️ Select at least one stem."

    files   = [Path(stem_paths_state[s]) for s in selected if s in stem_paths_state]
    missing = [s for s in selected if s not in stem_paths_state]

    if not files:
        return "", "⚠️ None of the selected stems were found in the last run."

    msg_parts = []
    if missing:
        msg_parts.append(f"(not generated: {', '.join(missing)} — skipped)")

    if len(files) == 1:
        msg_parts.insert(0, f"Only one stem — returning {files[0].name} as-is.")
        return _audio_html(str(files[0]), "Mix"), " ".join(msg_parts)

    out_dir = files[0].parent
    tag     = "_".join(sorted(selected))
    out     = out_dir / f"mix_{tag}.mp3"
    cmd     = ["ffmpeg", "-y"]
    for f in files:
        cmd += ["-i", str(f)]
    cmd += [
        "-filter_complex",
        f"amix=inputs={len(files)}:duration=longest:normalize=0",
        "-b:a", "320k",
        str(out),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        return "", f"❌ ffmpeg error:\n{e.stderr.decode()}"

    msg_parts.insert(0, f"✅ Mixed {len(files)} stems → `{out}`")
    return _audio_html(str(out), "Mix"), " ".join(msg_parts)


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

_best = PRESETS["best"]

with gr.Blocks(title="tabs-gen") as demo:

    gr.Markdown(
        "# 🎸 tabs-gen\n"
        "Split any audio into instrument stems — and optionally generate guitar tabs."
    )

    stems_dir_state = gr.State(value=None)

    with gr.Row(equal_height=False):

        # ------------------------------------------------------------------- #
        # LEFT — inputs + settings
        # ------------------------------------------------------------------- #
        with gr.Column(scale=1, min_width=340):

            gr.Markdown("### 🚀 Quick presets")
            with gr.Row():
                btn_fast     = gr.Button("⚡  Fast",         variant="secondary")
                btn_balanced = gr.Button("⭐  Balanced",     variant="secondary")
                btn_best     = gr.Button("🏆  Best Quality", variant="secondary")

            preset_desc = gr.Markdown(f"_{_best['desc']}_")

            gr.Markdown("---")

            audio_file = gr.Audio(
                label="Audio file (MP3 / WAV / FLAC / …)",
                type="filepath",
                sources=["upload"],
            )
            youtube_url = gr.Textbox(
                label="— or paste a YouTube URL —",
                placeholder="https://www.youtube.com/watch?v=…",
            )

            with gr.Accordion("⚙️  Settings", open=True):

                backend = gr.Radio(
                    choices=[
                        ("demucs  — faster, 4 or 6 stems",  "demucs"),
                        ("mdx  🏆 best quality, 4 stems",    "mdx"),
                    ],
                    value=_best["backend"],
                    label="Separation backend",
                )
                model = gr.Dropdown(
                    choices=[
                        ("htdemucs  — ⚡ fast, solid quality",        "htdemucs"),
                        ("htdemucs_ft  — ⭐ fine-tuned, best 4-stem", "htdemucs_ft"),
                        ("htdemucs_6s  — adds piano + other stems",   "htdemucs_6s"),
                    ],
                    value=_best["model"],
                    label="Demucs model",
                    info="Hidden when MDX backend is selected.",
                    visible=(_best["backend"] == "demucs"),
                )
                device = gr.Radio(
                    choices=[
                        ("mps  — Apple Silicon GPU ⚡ (recommended)", "mps"),
                        ("cuda — NVIDIA GPU",                          "cuda"),
                        ("cpu  — slowest, no GPU needed",              "cpu"),
                    ],
                    value="mps",
                    label="Device",
                )
                shifts = gr.Slider(
                    minimum=1, maximum=10, value=_best["shifts"], step=1,
                    label="Test-time shifts",
                    info="1 = fastest  ·  4 = balanced  ·  10 = best quality (slowest)",
                )
                generate_tabs = gr.Checkbox(
                    label="Generate tabs  (experimental — results are draft quality)",
                    value=False,
                )
                tab_warning = gr.Markdown(
                    "> ⚠️ **Tab generation is experimental.** Accuracy is limited — "
                    "expect ~65% for guitar, ~75% for bass, ~70% for drums. "
                    "Useful as a rough starting point, not a finished tab.",
                    visible=False,
                )
                with gr.Group(visible=False) as tab_options:
                    instruments = gr.CheckboxGroup(
                        choices=["guitar", "bass", "drums", "vocals"],
                        value=["guitar", "bass", "drums", "vocals"],
                        label="Instruments to transcribe",
                    )
                    formats = gr.CheckboxGroup(
                        choices=["ascii", "gp5"],
                        value=["ascii", "gp5"],
                        label="Tab output formats",
                    )
                keep_wav = gr.Checkbox(
                    label="Keep full-quality WAV stems alongside MP3s",
                    value=False,
                )

            with gr.Accordion("🔬  Advanced", open=False):
                onset_threshold = gr.Slider(0.0, 1.0, value=0.5, step=0.05,
                    label="Onset threshold (basic-pitch)")
                frame_threshold = gr.Slider(0.0, 1.0, value=0.3, step=0.05,
                    label="Frame threshold (basic-pitch)")
                crepe_model = gr.Dropdown(
                    choices=[
                        ("tiny   — ⚡ fastest",               "tiny"),
                        ("small",                              "small"),
                        ("medium — ⭐ balanced",              "medium"),
                        ("large",                              "large"),
                        ("full   — 🏆 best quality, slowest", "full"),
                    ],
                    value=_best["crepe_model"],
                    label="CREPE model (vocal pitch accuracy)",
                )
                output_dir = gr.Textbox(
                    label="Output directory",
                    value=str(Path.home() / "Music" / "tabs-gen"),
                )
                title_input = gr.Textbox(
                    label="Song title",
                    placeholder="Defaults to the audio filename stem",
                )

            with gr.Row():
                run_btn  = gr.Button("▶  Run",  variant="primary", size="lg")
                stop_btn = gr.Button("⏹  Stop", variant="stop",    size="lg")

            # Progress bar sits right below the buttons — always in view
            prog_bar = gr.HTML(value="", elem_id="tabs-gen-progress")

        # ------------------------------------------------------------------- #
        # RIGHT — outputs
        # ------------------------------------------------------------------- #
        with gr.Column(scale=1, min_width=400):

            logs = gr.Textbox(
                label="Pipeline log",
                lines=12,
                max_lines=12,
                autoscroll=True,
            )

            gr.Markdown("### 🎵 Stems")
            with gr.Row():
                stem_vocals = gr.HTML(_audio_html(None, "Vocals"))
                stem_guitar = gr.HTML(_audio_html(None, "Guitar"))
            with gr.Row():
                stem_bass  = gr.HTML(_audio_html(None, "Bass"))
                stem_drums = gr.HTML(_audio_html(None, "Drums"))
            with gr.Row():
                stem_piano = gr.HTML(_audio_html(None, "Piano"))
                stem_other = gr.HTML(_audio_html(None, "Other"))

            gr.Markdown("### 📄 Tab files")
            download_files = gr.Files(label="Download ASCII / GP5 tabs", interactive=False)

            gr.Markdown("### 🎛️ Custom Mix")
            gr.Markdown(
                "Select which stems to combine (or exclude). "
                "Available after a run completes."
            )
            mix_checks = gr.CheckboxGroup(
                choices=STEM_NAMES,
                value=["vocals", "guitar", "bass", "drums"],
                label="Stems to include in mix",
            )
            mix_btn    = gr.Button("🎚️  Create Mix", variant="secondary")
            mix_status = gr.Markdown("")
            mix_audio  = gr.HTML(_audio_html(None, "Mix"))

    # ----------------------------------------------------------------------- #
    # Wiring
    # ----------------------------------------------------------------------- #

    _preset_outputs = [backend, model, shifts, generate_tabs, crepe_model, preset_desc]
    btn_fast    .click(fn=lambda: _apply_preset("fast"),     outputs=_preset_outputs)
    btn_balanced.click(fn=lambda: _apply_preset("balanced"), outputs=_preset_outputs)
    btn_best    .click(fn=lambda: _apply_preset("best"),     outputs=_preset_outputs)

    generate_tabs.change(
        fn=lambda v: (gr.update(visible=v), gr.update(visible=v)),
        inputs=generate_tabs,
        outputs=[tab_warning, tab_options],
    )

    backend.change(
        fn=lambda b: gr.update(visible=(b == "demucs")),
        inputs=backend,
        outputs=model,
    )

    all_inputs = [
        audio_file, youtube_url, output_dir,
        backend, model, device, shifts,
        instruments, formats, generate_tabs, keep_wav,
        onset_threshold, frame_threshold, crepe_model,
        title_input,
    ]
    all_outputs = [
        logs,
        prog_bar,
        stem_vocals, stem_guitar, stem_bass, stem_drums, stem_piano, stem_other,
        download_files,
        stems_dir_state,
    ]

    run_event = run_btn.click(
        fn=_run,
        inputs=all_inputs,
        outputs=all_outputs,
    )
    stop_btn.click(fn=None, cancels=[run_event])

    mix_btn.click(
        fn=_create_mix,
        inputs=[stems_dir_state, mix_checks],
        outputs=[mix_audio, mix_status],
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    demo.queue()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        theme=gr.themes.Soft(),
        allowed_paths=[str(Path.home())],
    )
