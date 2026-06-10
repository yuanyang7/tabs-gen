"""Stage 4: ASCII tab renderer.

Renders guitar tabs, bass tabs, drum tabs, and vocal melody lines as
plain-text ASCII — the same format used on Ultimate Guitar and tab sites.
"""

from __future__ import annotations

from tabs_gen.stages.notation.bass import BassTab
from tabs_gen.stages.notation.drums import DRUM_LINES, DrumGrid
from tabs_gen.stages.notation.guitar import GuitarTab
from tabs_gen.stages.notation.vocals import VocalStaff

MEASURE_CELLS = 16      # 16th-note cells per measure (4/4 time)
LINE_WIDTH = 80         # characters per tab line before wrapping


def render_guitar_tab(tab: GuitarTab, cells_per_beat: int = 4) -> str:
    """Render a GuitarTab as multi-line ASCII tablature.

    The fret grid is laid out at 16th-note resolution.  Each note occupies
    the column closest to its onset time.

    Args:
        tab: output from notation.guitar.assign_frets()
        cells_per_beat: grid resolution (4 = 16th notes per beat)

    Returns:
        Multi-line ASCII string with one row per string.
    """
    if not tab.fretted_notes:
        return "(no guitar notes detected)\n"

    tempo = tab.tempo or 120.0
    beat_duration = 60.0 / tempo
    cell_duration = beat_duration / cells_per_beat

    max_time = max(fn.note.end for fn in tab.fretted_notes)
    total_cells = int(max_time / cell_duration) + cells_per_beat * 4

    num_strings = len(tab.tuning)
    # grid[string_idx][col] = fret string or "-"
    grid: list[list[str]] = [["-"] * total_cells for _ in range(num_strings)]

    for fn in tab.fretted_notes:
        col = int(round(fn.note.start / cell_duration))
        col = min(col, total_cells - 1)
        s = fn.string
        fret_str = str(fn.fret)
        # Write fret digits into consecutive cells
        for k, ch in enumerate(fret_str):
            if col + k < total_cells:
                grid[s][col + k] = ch
        # Pad cells after multi-digit fret numbers with dashes if neighbours are dashes
        after = col + len(fret_str)
        if after < total_cells and grid[s][after] == "-":
            pass  # already dash

    string_names = ["e", "B", "G", "D", "A", "E"]
    # Reverse so low E is at bottom
    lines: list[str] = []
    for si in reversed(range(num_strings)):
        name = string_names[si] if si < len(string_names) else str(si)
        row_cells = grid[si]
        lines.append(_format_tab_row(name, row_cells))

    return "\n".join(lines) + "\n"


def render_bass_tab(tab: BassTab, cells_per_beat: int = 4) -> str:
    """Render a BassTab as ASCII tablature."""
    if not tab.fretted_notes:
        return "(no bass notes detected)\n"

    tempo = tab.tempo or 120.0
    beat_duration = 60.0 / tempo
    cell_duration = beat_duration / cells_per_beat

    max_time = max(fn.note.end for fn in tab.fretted_notes)
    total_cells = int(max_time / cell_duration) + cells_per_beat * 4

    num_strings = len(tab.tuning)
    grid: list[list[str]] = [["-"] * total_cells for _ in range(num_strings)]

    for fn in tab.fretted_notes:
        col = int(round(fn.note.start / cell_duration))
        col = min(col, total_cells - 1)
        fret_str = str(fn.fret)
        for k, ch in enumerate(fret_str):
            if col + k < total_cells:
                grid[fn.string][col + k] = ch

    string_names = ["G", "D", "A", "E"]
    lines: list[str] = []
    for si in reversed(range(num_strings)):
        name = string_names[si] if si < len(string_names) else str(si)
        lines.append(_format_tab_row(name, grid[si]))

    return "\n".join(lines) + "\n"


def render_drum_tab(drum_grid: DrumGrid) -> str:
    """Render a DrumGrid as ASCII drum tablature."""
    if drum_grid.num_columns == 0:
        return "(no drum events detected)\n"

    lines: list[str] = []
    for line_name in DRUM_LINES:
        cells = drum_grid.grid.get(line_name, [])
        if not cells:
            continue
        # Only print lines that have at least one hit
        if all(c == "-" for c in cells):
            continue
        lines.append(_format_tab_row(line_name, cells))

    if not lines:
        return "(no drum events detected)\n"
    return "\n".join(lines) + "\n"


def render_vocal_staff(staff: VocalStaff) -> str:
    """Render a VocalStaff as a simple text melody line.

    Format:  note_name(duration_beats)  note_name(duration_beats) …
    Example: A4(1.0) G4(0.5) F#4(0.5) E4(2.0)
    """
    if not staff.notes:
        return "(no vocal notes detected)\n"

    tokens = []
    for vn in staff.notes:
        beats = round(vn.duration_beats * 4) / 4  # round to nearest quarter beat
        tokens.append(f"{vn.note_name}({beats:.2f})")

    # Wrap at LINE_WIDTH characters
    result_lines: list[str] = []
    current_line = "Melody: "
    for tok in tokens:
        if len(current_line) + len(tok) + 1 > LINE_WIDTH:
            result_lines.append(current_line)
            current_line = "        " + tok + " "
        else:
            current_line += tok + " "
    if current_line.strip():
        result_lines.append(current_line)

    return "\n".join(result_lines) + "\n"


def render_full_tab(
    guitar_tab: GuitarTab | None = None,
    bass_tab: BassTab | None = None,
    drum_grid: DrumGrid | None = None,
    vocal_staff: VocalStaff | None = None,
    title: str = "Generated Tab",
) -> str:
    """Combine all instrument tabs into a single text block."""
    sections: list[str] = []
    separator = "-" * 60

    sections.append(f"{'=' * 60}")
    sections.append(f"  {title}")
    sections.append(f"{'=' * 60}\n")

    if guitar_tab:
        sections.append("[ GUITAR ]")
        sections.append(render_guitar_tab(guitar_tab))
        sections.append(separator)

    if bass_tab:
        sections.append("[ BASS ]")
        sections.append(render_bass_tab(bass_tab))
        sections.append(separator)

    if drum_grid:
        sections.append("[ DRUMS ]")
        sections.append(render_drum_tab(drum_grid))
        sections.append(separator)

    if vocal_staff:
        sections.append("[ VOCALS ]")
        sections.append(render_vocal_staff(vocal_staff))
        sections.append(separator)

    return "\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_tab_row(name: str, cells: list[str]) -> str:
    """Format a single tab row, wrapping at LINE_WIDTH characters.

    Inserts bar lines every MEASURE_CELLS columns.
    """
    header = f"{name:>2}|"
    content = ""
    chunks: list[str] = []
    col = 0

    for i, cell in enumerate(cells):
        if i > 0 and i % MEASURE_CELLS == 0:
            content += "|"
        content += cell

    content += "|"

    # Wrap long rows
    max_content_width = LINE_WIDTH - len(header)
    lines = []
    while len(content) > max_content_width:
        # Find a clean break at a bar line
        split_at = content.rfind("|", 0, max_content_width)
        if split_at <= 0:
            split_at = max_content_width
        lines.append(header + content[:split_at + 1])
        content = content[split_at + 1:]

    if content:
        lines.append(header + content)

    return "\n".join(lines)
