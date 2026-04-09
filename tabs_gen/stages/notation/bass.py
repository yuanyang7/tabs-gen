"""Stage 3: Bass guitar tab assignment.

Bass is largely monophonic, so the DP is simpler than guitar: we just
minimise hand position movement across successive single notes.

Standard 4-string bass tuning (low → high): E1 A1 D2 G2
MIDI open notes:                             28 33 38 43
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tabs_gen.utils.midi_utils import MidiData, Note

BASS_TUNING: list[tuple[int, int]] = [
    (0, 28),  # E1
    (1, 33),  # A1
    (2, 38),  # D2
    (3, 43),  # G2
]
NUM_FRETS = 24
POSITION_SHIFT_COST = 1.0


@dataclass
class BassFrettedNote:
    note: Note
    string: int   # 0 = low E
    fret: int

    def string_name(self) -> str:
        return ["E", "A", "D", "G"][self.string] if self.string < 4 else str(self.string)


@dataclass
class BassTab:
    fretted_notes: list[BassFrettedNote] = field(default_factory=list)
    tempo: float = 120.0
    tuning: list[tuple[int, int]] = field(default_factory=lambda: list(BASS_TUNING))

    def string_name(self, idx: int) -> str:
        names = ["E", "A", "D", "G"]
        return names[idx] if idx < len(names) else str(idx)


def assign_frets(midi_data: MidiData, tuning: list[tuple[int, int]] | None = None) -> BassTab:
    """Assign string/fret positions to bass notes using DP.

    Prefers lower strings (thicker tone) and lower fret positions
    as tiebreakers, which matches standard bass playing convention.
    """
    if tuning is None:
        tuning = list(BASS_TUNING)

    notes = sorted(midi_data.notes, key=lambda n: n.start)
    if not notes:
        return BassTab(tempo=midi_data.tempo, tuning=tuning)

    fretted = _dp_assign(notes, tuning)
    return BassTab(fretted_notes=fretted, tempo=midi_data.tempo, tuning=tuning)


def _valid_positions(midi_pitch: int, tuning: list[tuple[int, int]]) -> list[tuple[int, int]]:
    positions = []
    for string_idx, open_midi in tuning:
        fret = midi_pitch - open_midi
        if 0 <= fret <= NUM_FRETS:
            positions.append((string_idx, fret))
    # Prefer lower strings first (tiebreaker: lower fret on lower string)
    return sorted(positions, key=lambda p: (p[1], p[0]))


def _dp_assign(notes: list[Note], tuning: list[tuple[int, int]]) -> list[BassFrettedNote]:
    """DP: minimise total hand position shift across the note sequence."""
    INF = float("inf")
    n = len(notes)
    candidates = [_valid_positions(note.pitch, tuning) for note in notes]

    # Replace empty candidate lists with a placeholder (pitch out of range)
    for i, c in enumerate(candidates):
        if not c:
            candidates[i] = [(0, 0)]

    # dp[i][p] = (min_cost, prev_position_idx)
    dp: list[list[tuple[float, int]]] = [
        [(INF, -1)] * len(candidates[i]) for i in range(n)
    ]

    for p in range(len(candidates[0])):
        dp[0][p] = (float(candidates[0][p][1]), -1)  # initial cost = fret number

    for i in range(1, n):
        for p, (_, fret) in enumerate(candidates[i]):
            best_cost = INF
            best_prev = 0
            for pp, (_, prev_fret) in enumerate(candidates[i - 1]):
                cost = dp[i - 1][pp][0] + abs(fret - prev_fret) * POSITION_SHIFT_COST
                if cost < best_cost:
                    best_cost = cost
                    best_prev = pp
            dp[i][p] = (best_cost, best_prev)

    # Backtrack
    best_last = min(range(len(dp[n - 1])), key=lambda p: dp[n - 1][p][0])
    path: list[int] = [best_last]
    for i in range(n - 1, 0, -1):
        path.append(dp[i][path[-1]][1])
    path.reverse()

    return [
        BassFrettedNote(note=notes[i], string=candidates[i][path[i]][0], fret=candidates[i][path[i]][1])
        for i in range(n)
    ]
