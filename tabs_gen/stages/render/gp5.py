"""Stage 4: Guitar Pro 5 (.gp5) file writer via PyGuitarPro.

Writes guitar, bass, and drum tracks into a single .gp5 file that can be
opened in Guitar Pro 5+, TuxGuitar (free), or MuseScore (with plugin).
"""

from __future__ import annotations

import logging
from pathlib import Path

from tabs_gen.stages.notation.bass import BassTab
from tabs_gen.stages.notation.drums import DrumGrid, DRUM_LINES
from tabs_gen.stages.notation.guitar import GuitarTab

logger = logging.getLogger(__name__)

# GP timing constants
QUARTER_TIME = 960          # ticks per quarter note (guitarpro.Duration.quarterTime)
CELLS_PER_BEAT = 4          # 16th-note grid
CELL_TICKS = QUARTER_TIME // CELLS_PER_BEAT   # 240 ticks per 16th note
CELLS_PER_MEASURE = 16      # 4/4 at 16th-note resolution
MEASURE_TICKS = QUARTER_TIME * 4              # 3840 ticks per 4/4 measure
FIRST_MEASURE_START = QUARTER_TIME            # GP5 convention: measure 1 starts at tick 960

# General MIDI drum note mapping (line_name → GM note)
LINE_TO_GM: dict[str, int] = {
    "CC": 49,  "RC": 51,  "HH": 42,  "HO": 46,
    "S ": 38,  "T1": 48,  "T2": 45,  "T3": 41,  "BD": 36,
}


def write_gp5(
    output_path: str | Path,
    guitar_tab: GuitarTab | None = None,
    bass_tab: BassTab | None = None,
    drum_grid: DrumGrid | None = None,
    title: str = "Generated Tab",
    tempo: float = 120.0,
) -> Path:
    """Write instrument data to a Guitar Pro 5 file."""
    try:
        import guitarpro  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pyguitarpro is required for GP5 output. "
            "Install it with: pip install pyguitarpro"
        ) from e

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # GP5 only supports Latin-1 in string fields
    safe_title = title.encode("latin-1", errors="ignore").decode("latin-1") or "Generated Tab"

    # ── Compute beat/cell timing from actual tempo ────────────────────────────
    beat_duration_sec = 60.0 / (tempo or 120.0)
    cell_duration_sec = beat_duration_sec / CELLS_PER_BEAT

    def time_to_cell(t: float) -> int:
        return int(round(t / cell_duration_sec))

    def cell_to_tick(cell: int) -> int:
        """Convert 16th-note cell index to absolute GP tick."""
        return FIRST_MEASURE_START + cell * CELL_TICKS

    # ── Determine total measures needed ──────────────────────────────────────
    max_cell = 0
    if guitar_tab and guitar_tab.fretted_notes:
        max_cell = max(max_cell, time_to_cell(max(fn.note.end for fn in guitar_tab.fretted_notes)))
    if bass_tab and bass_tab.fretted_notes:
        max_cell = max(max_cell, time_to_cell(max(fn.note.end for fn in bass_tab.fretted_notes)))
    if drum_grid:
        max_cell = max(max_cell, drum_grid.num_columns)
    num_measures = max(1, (max_cell + CELLS_PER_MEASURE) // CELLS_PER_MEASURE + 1)

    # ── Build Song + shared MeasureHeaders ───────────────────────────────────
    song = guitarpro.Song()
    song.title = safe_title
    song.tempo = int(tempo)
    song.tracks = []

    # Build all measure headers (Song() already has 1; replace it)
    headers: list = []
    for i in range(num_measures):
        h = guitarpro.MeasureHeader()
        h.number = i + 1
        h.start = FIRST_MEASURE_START + i * MEASURE_TICKS
        h.timeSignature.numerator = 4
        h.timeSignature.denominator.value = 4
        headers.append(h)
    song.measureHeaders = headers

    # ── Build tracks ─────────────────────────────────────────────────────────
    track_number = 1
    if guitar_tab:
        cell_map = _notes_to_cell_map(
            sorted(guitar_tab.fretted_notes, key=lambda fn: fn.note.start),
            time_to_cell,
        )
        t = _make_track(song, guitarpro, track_number, "Guitar", 25,
                        [(i + 1, midi) for i, (_, midi) in enumerate(guitar_tab.tuning)],
                        headers, cell_map, cell_to_tick, is_drum=False)
        song.tracks.append(t)
        track_number += 1

    if bass_tab:
        cell_map = _notes_to_cell_map(
            sorted(bass_tab.fretted_notes, key=lambda fn: fn.note.start),
            time_to_cell,
        )
        t = _make_track(song, guitarpro, track_number, "Bass", 33,
                        [(i + 1, midi) for i, (_, midi) in enumerate(bass_tab.tuning)],
                        headers, cell_map, cell_to_tick, is_drum=False)
        song.tracks.append(t)
        track_number += 1

    if drum_grid:
        t = _make_drum_track(song, guitarpro, track_number, drum_grid, headers, cell_to_tick)
        song.tracks.append(t)
        track_number += 1

    if not song.tracks:
        logger.warning("No tracks to write — adding empty placeholder")
        t = guitarpro.Track(song, number=1)
        t.measures = [guitarpro.Measure(t, h) for h in headers]
        song.tracks.append(t)

    guitarpro.write(song, str(output_path))
    logger.info("Wrote GP5 file: %s", output_path)
    return output_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _quantize_velocity(gp, midi_velocity: int) -> int:
    """Map 0-127 MIDI velocity to nearest Guitar Pro velocity step."""
    v = gp.Velocities
    steps = [v.pianoPianissimo, v.pianissimo, v.piano, v.mezzoPiano,
             v.mezzoForte, v.forte, v.fortissimo, v.forteFortissimo]
    return min(steps, key=lambda s: abs(s - midi_velocity))


def _notes_to_cell_map(fretted_notes, time_to_cell) -> dict[int, list]:
    """Map 16th-note cell index → list of fretted notes."""
    cell_map: dict[int, list] = {}
    for fn in fretted_notes:
        cell = time_to_cell(fn.note.start)
        cell_map.setdefault(cell, []).append(fn)
    return cell_map


def _make_track(song, gp, number: int, name: str, instrument: int,
                strings: list[tuple[int, int]], headers: list,
                cell_map: dict[int, list], cell_to_tick, is_drum: bool):
    track = gp.Track(song, number=number)
    track.name = name
    track.channel.instrument = instrument
    track.strings = [gp.GuitarString(num, midi) for num, midi in strings]
    track.fretCount = 24

    measures = []
    for i, header in enumerate(headers):
        measure = gp.Measure(track, header)
        voice = measure.voices[0]
        beats = []

        for cell in range(CELLS_PER_MEASURE):
            global_cell = i * CELLS_PER_MEASURE + cell
            tick = cell_to_tick(global_cell)
            notes_at_cell = cell_map.get(global_cell, [])

            beat = gp.Beat(voice)
            beat.start = tick
            beat.duration.value = 16  # 16th note

            if notes_at_cell:
                # GP5 allows at most one note per string per beat; deduplicate
                seen_strings: set[int] = set()
                for fn in notes_at_cell:
                    gp_string = fn.string + 1  # convert to 1-indexed
                    if gp_string in seen_strings:
                        continue
                    seen_strings.add(gp_string)
                    note = gp.Note(beat)
                    note.string = gp_string
                    note.value = fn.fret
                    note.type = gp.NoteType.normal
                    note.velocity = _quantize_velocity(gp, int(getattr(fn.note, "velocity", 95)))
                    beat.notes.append(note)
                beat.status = gp.BeatStatus.normal
            else:
                beat.status = gp.BeatStatus.rest

            beats.append(beat)

        voice.beats = beats
        measures.append(measure)

    track.measures = measures
    return track


def _make_drum_track(song, gp, number: int, drum_grid: DrumGrid,
                     headers: list, cell_to_tick):
    track = gp.Track(song, number=number)
    track.name = "Drums"
    track.channel.channel = 9
    track.channel.effectChannel = 9
    track.channel.instrument = 0
    track.isPercussionTrack = True
    track.strings = [gp.GuitarString(i + 1, 36) for i in range(6)]

    num_cells = drum_grid.num_columns
    measures = []

    for i, header in enumerate(headers):
        measure = gp.Measure(track, header)
        voice = measure.voices[0]
        beats = []

        for cell in range(CELLS_PER_MEASURE):
            global_cell = i * CELLS_PER_MEASURE + cell
            tick = cell_to_tick(global_cell)

            beat = gp.Beat(voice)
            beat.start = tick
            beat.duration.value = 16

            drum_notes = []
            if global_cell < num_cells:
                for line_name in DRUM_LINES:
                    symbols = drum_grid.grid.get(line_name, [])
                    if global_cell < len(symbols) and symbols[global_cell] != "-":
                        gm_note = LINE_TO_GM.get(line_name.strip(), 36)
                        note = gp.Note(beat)
                        note.string = 1
                        note.value = gm_note
                        note.type = gp.NoteType.normal
                        note.velocity = 95
                        drum_notes.append(note)

            if drum_notes:
                beat.notes = drum_notes
                beat.status = gp.BeatStatus.normal
            else:
                beat.status = gp.BeatStatus.rest

            beats.append(beat)

        voice.beats = beats
        measures.append(measure)

    track.measures = measures
    return track
