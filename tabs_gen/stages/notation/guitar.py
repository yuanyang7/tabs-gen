"""Stage 3: Guitar fret/string assignment via dynamic programming.

Given a sequence of MIDI notes, finds a (string, fret) assignment for each
note that minimises total hand-position travel while respecting playability
constraints (max 4-fret stretch, no two simultaneous notes on same string).

Standard guitar tuning (low → high): E2 A2 D3 G3 B3 E4
MIDI open notes:                      40 45 50 55 59 64
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tabs_gen.utils.midi_utils import MidiData, Note

# (string_index, open_midi_note)  — string 0 = low E
STANDARD_TUNING: list[tuple[int, int]] = [
    (0, 40),  # E2
    (1, 45),  # A2
    (2, 50),  # D3
    (3, 55),  # G3
    (4, 59),  # B3
    (5, 64),  # E4
]
NUM_FRETS = 24
MAX_STRETCH = 4  # maximum fret span across all fingers at one moment
POSITION_SHIFT_COST = 1.0  # cost per fret of position change between beats
STRETCH_COST = 0.5          # cost per fret beyond MAX_STRETCH


@dataclass
class FrettedNote:
    """A MIDI note assigned to a specific string and fret."""
    note: Note
    string: int   # 0 = low E, 5 = high E
    fret: int     # 0 = open string

    @property
    def display_fret(self) -> str:
        return str(self.fret)


@dataclass
class GuitarTab:
    """Sequence of fretted notes with tempo metadata."""
    fretted_notes: list[FrettedNote] = field(default_factory=list)
    tempo: float = 120.0
    tuning: list[tuple[int, int]] = field(default_factory=lambda: list(STANDARD_TUNING))

    def string_name(self, idx: int) -> str:
        names = ["E", "A", "D", "G", "B", "e"]
        return names[idx] if idx < len(names) else str(idx)


def assign_frets(midi_data: MidiData, tuning: list[tuple[int, int]] | None = None) -> GuitarTab:
    """Assign string/fret positions to each note in midi_data using DP.

    The algorithm processes groups of simultaneous notes (chords) as atomic
    units, finding the chord voicing that minimises the cost function:

        cost = position_shift + stretch_penalty

    Args:
        midi_data: MIDI notes from the guitar stem transcription.
        tuning: list of (string_idx, open_midi) pairs; defaults to standard.

    Returns:
        GuitarTab with each note assigned a (string, fret).
    """
    if tuning is None:
        tuning = list(STANDARD_TUNING)

    notes = list(midi_data.notes)
    if not notes:
        return GuitarTab(tempo=midi_data.tempo, tuning=tuning)

    # Group notes into simultaneous chords (notes overlapping in time)
    chord_groups = _group_into_chords(notes)

    # For each chord, compute all valid voicings (one fret per note, no string collision)
    chord_voicings: list[list[list[tuple[int, int]]]] = []
    for chord in chord_groups:
        voicings = _chord_voicings(chord, tuning)
        if not voicings:
            # No valid voicing — fall back to any single-note assignment
            voicings = _fallback_voicings(chord, tuning)
        chord_voicings.append(voicings)

    # DP: find the path through voicings minimising total hand position cost
    best_path = _dp_solve(chord_voicings)

    # Flatten into FrettedNote list
    fretted: list[FrettedNote] = []
    for chord, voicing in zip(chord_groups, best_path):
        for note, (string, fret) in zip(chord, voicing):
            fretted.append(FrettedNote(note=note, string=string, fret=fret))

    fretted.sort(key=lambda fn: fn.note.start)
    return GuitarTab(fretted_notes=fretted, tempo=midi_data.tempo, tuning=tuning)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _group_into_chords(notes: list[Note], tolerance: float = 0.05) -> list[list[Note]]:
    """Group notes that start within `tolerance` seconds of each other."""
    if not notes:
        return []

    sorted_notes = sorted(notes, key=lambda n: n.start)
    groups: list[list[Note]] = [[sorted_notes[0]]]

    for note in sorted_notes[1:]:
        if note.start - groups[-1][0].start <= tolerance:
            groups[-1].append(note)
        else:
            groups.append([note])

    return groups


def _valid_positions(midi_pitch: int, tuning: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return all (string, fret) positions that can play `midi_pitch`."""
    positions = []
    for string_idx, open_midi in tuning:
        fret = midi_pitch - open_midi
        if 0 <= fret <= NUM_FRETS:
            positions.append((string_idx, fret))
    return positions


def _chord_voicings(
    chord: list[Note],
    tuning: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """Return all valid voicings for a chord (no string collisions).

    A voicing is a list of (string, fret) pairs, one per note in `chord`.
    """
    # Build candidate positions per note
    candidates = [_valid_positions(n.pitch, tuning) for n in chord]

    # Enumerate all combinations — prune for string collisions and excessive stretch
    voicings: list[list[tuple[int, int]]] = []
    _enumerate_voicings(candidates, [], set(), voicings)
    return voicings


def _enumerate_voicings(
    remaining: list[list[tuple[int, int]]],
    current: list[tuple[int, int]],
    used_strings: set[int],
    result: list[list[tuple[int, int]]],
    max_results: int = 200,
) -> None:
    if not remaining:
        result.append(list(current))
        return
    if len(result) >= max_results:
        return

    for string, fret in remaining[0]:
        if string in used_strings:
            continue
        new_current = current + [(string, fret)]
        # Check stretch constraint only for fretted (non-open) notes
        fretted_frets = [f for _, f in new_current if f > 0]
        if len(fretted_frets) >= 2:
            stretch = max(fretted_frets) - min(fretted_frets)
            if stretch > MAX_STRETCH + 1:
                continue
        _enumerate_voicings(
            remaining[1:],
            new_current,
            used_strings | {string},
            result,
            max_results,
        )


def _fallback_voicings(chord: list[Note], tuning: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """Assign each note to its lowest available position, ignoring string collisions."""
    voicing = []
    for note in chord:
        positions = _valid_positions(note.pitch, tuning)
        if positions:
            voicing.append(sorted(positions, key=lambda p: p[1])[0])  # lowest fret
        else:
            voicing.append((0, 0))  # placeholder
    return [voicing]


def _dp_solve(
    chord_voicings: list[list[list[tuple[int, int]]]],
) -> list[list[tuple[int, int]]]:
    """Dynamic programming search for the minimum-cost voicing sequence.

    State: index of chosen voicing at each chord step.
    Cost: hand position shift between consecutive chords.
    """
    n = len(chord_voicings)
    if n == 0:
        return []

    # dp[i][v] = (min_cost, prev_voicing_idx)
    INF = float("inf")
    dp: list[list[tuple[float, int]]] = [
        [(INF, -1)] * len(chord_voicings[i]) for i in range(n)
    ]

    # Initialise first chord
    for v, voicing in enumerate(chord_voicings[0]):
        dp[0][v] = (_voicing_cost(voicing), -1)

    # Forward pass
    for i in range(1, n):
        for v, voicing in enumerate(chord_voicings[i]):
            best_cost = INF
            best_prev = 0
            for pv, prev_voicing in enumerate(chord_voicings[i - 1]):
                transition = _transition_cost(prev_voicing, voicing)
                total = dp[i - 1][pv][0] + transition
                if total < best_cost:
                    best_cost = total
                    best_prev = pv
            dp[i][v] = (best_cost + _voicing_cost(voicing), best_prev)

    # Backtrack
    best_last = min(range(len(dp[n - 1])), key=lambda v: dp[n - 1][v][0])
    path_indices = [best_last]
    for i in range(n - 1, 0, -1):
        path_indices.append(dp[i][path_indices[-1]][1])
    path_indices.reverse()

    return [chord_voicings[i][v] for i, v in enumerate(path_indices)]


def _voicing_cost(voicing: list[tuple[int, int]]) -> float:
    """Internal cost of a voicing — penalises excessive stretch and high frets.

    A small per-fret penalty (0.01) acts as a tiebreaker that prefers open
    strings and lower positions over equivalent high-fret positions.
    """
    fretted = [f for _, f in voicing if f > 0]
    stretch_penalty = 0.0
    if len(fretted) >= 2:
        stretch = max(fretted) - min(fretted)
        stretch_penalty = max(0.0, stretch - MAX_STRETCH) * STRETCH_COST
    # Tiebreaker: prefer lower fret positions (open strings are free)
    fret_bias = sum(f * 0.01 for _, f in voicing)
    return stretch_penalty + fret_bias


def _transition_cost(prev: list[tuple[int, int]], curr: list[tuple[int, int]]) -> float:
    """Cost of moving from one chord voicing to the next.

    Measured as the absolute change in the lowest fretted position
    (i.e., the hand's anchor point on the neck).
    """
    def anchor(voicing: list[tuple[int, int]]) -> int:
        frets = [f for _, f in voicing if f > 0]
        return min(frets) if frets else 0

    return abs(anchor(curr) - anchor(prev)) * POSITION_SHIFT_COST
