"""Stage 3: Drum notation mapping.

Maps drum onsets to a standard ASCII drum tab grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tabs_gen.stages.transcription import DrumData, DrumOnset

# Lines in the ASCII drum tab, top to bottom
DRUM_LINES = ["CC", "RC", "HH", "HO", "S ", "T1", "T2", "T3", "BD"]

PART_TO_LINE: dict[str, str] = {
    "crash":          "CC",
    "ride":           "RC",
    "hi_hat_closed":  "HH",
    "hi_hat_pedal":   "HH",
    "hi_hat_open":    "HO",
    "snare":          "S ",
    "tom_high":       "T1",
    "tom_mid":        "T2",
    "tom_low":        "T3",
    "kick":           "BD",
}

# ASCII symbol used for each drum line
LINE_SYMBOL: dict[str, str] = {
    "CC": "X",  # crash — hard accent
    "RC": "x",  # ride
    "HH": "x",  # hi-hat closed
    "HO": "o",  # hi-hat open
    "S ": "o",  # snare
    "T1": "o", "T2": "o", "T3": "o",  # toms
    "BD": "o",  # bass drum
}


@dataclass
class DrumGrid:
    """A quantised grid of drum hits ready for rendering."""
    # grid[line][column] → symbol or "-"
    grid: dict[str, list[str]] = field(default_factory=dict)
    num_columns: int = 0
    tempo: float = 120.0
    grid_resolution: int = 16  # 16th-note grid

    def __post_init__(self) -> None:
        if not self.grid:
            self.grid = {line: [] for line in DRUM_LINES}


def build_drum_grid(
    drum_data: DrumData,
    grid_resolution: int = 16,
    bars: int | None = None,
) -> DrumGrid:
    """Quantise drum onsets onto a rhythmic grid.

    Args:
        drum_data: onset events from the transcription stage.
        grid_resolution: subdivisions per beat (16 = 16th notes).
        bars: number of 4/4 bars to render; auto-detected if None.

    Returns:
        DrumGrid ready for ASCII rendering.
    """
    tempo = drum_data.tempo or 120.0
    beat_duration = 60.0 / tempo
    cell_duration = beat_duration / (grid_resolution / 4)  # duration per grid cell

    if not drum_data.onsets:
        total_cells = (bars or 2) * grid_resolution * 4
        grid = {line: ["-"] * total_cells for line in DRUM_LINES}
        return DrumGrid(grid=grid, num_columns=total_cells, tempo=tempo, grid_resolution=grid_resolution)

    max_time = max(o.time for o in drum_data.onsets)
    total_beats = max_time / beat_duration + 4  # add 4-beat padding
    if bars is not None:
        total_beats = max(total_beats, bars * 4)
    total_cells = int(total_beats * grid_resolution / 4) + 1

    grid: dict[str, list[str]] = {line: ["-"] * total_cells for line in DRUM_LINES}

    for onset in drum_data.onsets:
        line = PART_TO_LINE.get(onset.part)
        if line is None:
            continue
        col = int(round(onset.time / cell_duration))
        col = min(col, total_cells - 1)
        symbol = LINE_SYMBOL.get(line, "x")
        # hi-hat open overrides closed on same cell
        if line == "HH" and onset.part == "hi_hat_open":
            symbol = LINE_SYMBOL["HO"]
            grid["HO"][col] = symbol
        else:
            grid[line][col] = symbol

    return DrumGrid(grid=grid, num_columns=total_cells, tempo=tempo, grid_resolution=grid_resolution)
