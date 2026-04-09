"""MIDI helpers: loading, quantization, and note extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Note:
    """A single musical note with timing information."""
    pitch: int          # MIDI pitch (0-127)
    start: float        # onset in seconds
    end: float          # offset in seconds
    velocity: int = 80

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class MidiData:
    """Simplified MIDI representation used throughout the pipeline."""
    notes: list[Note] = field(default_factory=list)
    tempo: float = 120.0        # BPM
    time_signature: tuple[int, int] = (4, 4)
    instrument_name: str = "unknown"

    def sort(self) -> None:
        self.notes.sort(key=lambda n: n.start)


def load_midi(path: str | Path) -> MidiData:
    """Load a MIDI file into a MidiData object using pretty_midi."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(path))
    tempo_change_times, tempos = pm.get_tempo_change_times(), pm.get_tempo_change_times()
    tempo = float(pm.estimate_tempo()) if pm.instruments else 120.0

    notes: list[Note] = []
    instrument_name = "unknown"
    for inst in pm.instruments:
        if not inst.is_drum:
            instrument_name = pretty_midi.program_to_instrument_name(inst.program)
        for n in inst.notes:
            notes.append(Note(pitch=n.pitch, start=n.start, end=n.end, velocity=n.velocity))

    md = MidiData(notes=notes, tempo=tempo, instrument_name=instrument_name)
    md.sort()
    return md


def quantize_notes(notes: list[Note], tempo: float, resolution: int = 16) -> list[Note]:
    """Snap note onsets and offsets to the nearest rhythmic grid.

    Args:
        notes: list of Note objects with continuous timing.
        tempo: BPM used to calculate grid spacing.
        resolution: grid subdivisions per beat (16 = 16th notes).

    Returns:
        New list of Note objects with quantized start/end times.
    """
    beat_duration = 60.0 / tempo
    grid_size = beat_duration / (resolution / 4)  # quarter-note grid unit

    def snap(t: float) -> float:
        return round(t / grid_size) * grid_size

    quantized = []
    for n in notes:
        qs = snap(n.start)
        qe = snap(n.end)
        if qe <= qs:
            qe = qs + grid_size
        quantized.append(Note(pitch=n.pitch, start=qs, end=qe, velocity=n.velocity))
    return quantized


def pitch_contour_to_notes(
    times: np.ndarray,
    frequencies: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.5,
    min_duration: float = 0.05,
) -> list[Note]:
    """Convert a continuous pitch contour (from CREPE/pYIN) to discrete notes.

    Segments the contour into note regions by detecting unvoiced gaps and
    significant pitch jumps (>= 1 semitone from running mean).

    Args:
        times: frame timestamps in seconds.
        frequencies: pitch in Hz per frame (0 = unvoiced).
        confidence: per-frame confidence from pitch tracker.
        confidence_threshold: frames below this are treated as unvoiced.
        min_duration: notes shorter than this (seconds) are discarded.

    Returns:
        List of Note objects with MIDI pitches.
    """
    voiced = (confidence >= confidence_threshold) & (frequencies > 0)
    notes: list[Note] = []

    i = 0
    while i < len(times):
        if not voiced[i]:
            i += 1
            continue

        # start of a voiced segment
        seg_start = i
        pitches_hz: list[float] = []

        while i < len(times) and voiced[i]:
            pitches_hz.append(float(frequencies[i]))
            i += 1

        seg_end = i - 1
        if len(pitches_hz) == 0:
            continue

        start_time = float(times[seg_start])
        end_time = float(times[seg_end])
        if end_time - start_time < min_duration:
            continue

        # Median frequency → MIDI pitch
        median_hz = float(np.median(pitches_hz))
        if median_hz <= 0:
            continue
        midi_pitch = int(round(69 + 12 * math.log2(median_hz / 440.0)))
        midi_pitch = max(0, min(127, midi_pitch))

        notes.append(Note(pitch=midi_pitch, start=start_time, end=end_time))

    return notes


def hz_to_midi(freq: float) -> int:
    """Convert a frequency in Hz to the nearest MIDI note number."""
    if freq <= 0:
        return 0
    return int(round(69 + 12 * math.log2(freq / 440.0)))


def midi_to_note_name(midi: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[midi % 12]}{midi // 12 - 1}"
