"""Stage 3: Vocal notation — quantise pitch contour to a note staff.

Produces a simple list of (note_name, duration_beats) pairs suitable
for rendering as text or passing to music21 for MusicXML export.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tabs_gen.utils.midi_utils import MidiData, Note, midi_to_note_name, quantize_notes


@dataclass
class VocalNote:
    """A single vocal note for display."""
    pitch: int          # MIDI pitch
    note_name: str      # e.g. "A4", "C#5"
    start: float        # quantized start in seconds
    duration: float     # quantized duration in seconds
    duration_beats: float  # duration in beats


@dataclass
class VocalStaff:
    """Quantised vocal melody ready for rendering."""
    notes: list[VocalNote] = field(default_factory=list)
    tempo: float = 120.0


def build_vocal_staff(midi_data: MidiData) -> VocalStaff:
    """Quantise vocal notes and annotate them with note names and beat durations."""
    tempo = midi_data.tempo or 120.0
    beat_duration = 60.0 / tempo

    quantized = quantize_notes(midi_data.notes, tempo=tempo)

    vocal_notes = []
    for note in quantized:
        dur_beats = note.duration / beat_duration
        vocal_notes.append(VocalNote(
            pitch=note.pitch,
            note_name=midi_to_note_name(note.pitch),
            start=note.start,
            duration=note.duration,
            duration_beats=dur_beats,
        ))

    return VocalStaff(notes=vocal_notes, tempo=tempo)
