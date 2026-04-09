"""Unit tests for notation stages — no ML dependencies required."""

import pytest

from tabs_gen.utils.midi_utils import Note, MidiData, quantize_notes, hz_to_midi
from tabs_gen.stages.notation.guitar import assign_frets as guitar_assign_frets, STANDARD_TUNING
from tabs_gen.stages.notation.bass import assign_frets as bass_assign_frets, BASS_TUNING
from tabs_gen.stages.notation.drums import build_drum_grid
from tabs_gen.stages.notation.vocals import build_vocal_staff
from tabs_gen.stages.transcription import DrumData, DrumOnset
from tabs_gen.stages.output.ascii_tab import (
    render_guitar_tab,
    render_bass_tab,
    render_drum_tab,
    render_vocal_staff,
    render_full_tab,
)


# ---------------------------------------------------------------------------
# midi_utils
# ---------------------------------------------------------------------------

def test_hz_to_midi_a4():
    assert hz_to_midi(440.0) == 69  # A4

def test_hz_to_midi_middle_c():
    assert hz_to_midi(261.63) == 60  # C4

def test_quantize_notes_snaps_to_grid():
    notes = [Note(pitch=60, start=0.02, end=0.27)]  # slightly off grid
    quantized = quantize_notes(notes, tempo=120.0, resolution=16)
    # At 120 BPM, 16th note = 0.125s. start 0.02 → snaps to 0.0, end 0.27 → 0.25
    assert quantized[0].start == pytest.approx(0.0, abs=0.01)
    assert quantized[0].end == pytest.approx(0.25, abs=0.02)


# ---------------------------------------------------------------------------
# Guitar fret/string assignment
# ---------------------------------------------------------------------------

def test_guitar_open_e():
    # MIDI 64 = E4 = string 1 open (fret 0) in standard tuning
    midi = MidiData(notes=[Note(pitch=64, start=0.0, end=0.5)], tempo=120.0)
    tab = guitar_assign_frets(midi)
    assert len(tab.fretted_notes) == 1
    fn = tab.fretted_notes[0]
    assert fn.fret == 0
    assert fn.string == 5  # highest string (e)

def test_guitar_c_major_scale():
    # C major scale on guitar: C4(60) D4(62) E4(64) F4(65) G4(67) A4(69) B4(71) C5(72)
    notes = [
        Note(pitch=p, start=i * 0.5, end=i * 0.5 + 0.4)
        for i, p in enumerate([60, 62, 64, 65, 67, 69, 71, 72])
    ]
    midi = MidiData(notes=notes, tempo=120.0)
    tab = guitar_assign_frets(midi)
    assert len(tab.fretted_notes) == 8
    # All frets should be 0–24
    for fn in tab.fretted_notes:
        assert 0 <= fn.fret <= 24
        assert 0 <= fn.string <= 5

def test_guitar_empty_midi():
    midi = MidiData(notes=[], tempo=120.0)
    tab = guitar_assign_frets(midi)
    assert tab.fretted_notes == []


# ---------------------------------------------------------------------------
# Bass fret/string assignment
# ---------------------------------------------------------------------------

def test_bass_open_e():
    # MIDI 28 = E1 = low E string open (fret 0)
    midi = MidiData(notes=[Note(pitch=28, start=0.0, end=0.5)], tempo=120.0)
    tab = bass_assign_frets(midi)
    assert tab.fretted_notes[0].fret == 0
    assert tab.fretted_notes[0].string == 0

def test_bass_ascending_line():
    notes = [
        Note(pitch=p, start=i * 0.25, end=i * 0.25 + 0.2)
        for i, p in enumerate([28, 30, 33, 35, 38, 40, 43, 45])
    ]
    midi = MidiData(notes=notes, tempo=120.0)
    tab = bass_assign_frets(midi)
    assert len(tab.fretted_notes) == 8
    for fn in tab.fretted_notes:
        assert 0 <= fn.fret <= 24


# ---------------------------------------------------------------------------
# Drum grid
# ---------------------------------------------------------------------------

def test_drum_grid_basic():
    onsets = [
        DrumOnset(time=0.0, part="kick"),
        DrumOnset(time=0.5, part="snare"),
        DrumOnset(time=0.0, part="hi_hat_closed"),
        DrumOnset(time=0.25, part="hi_hat_closed"),
    ]
    drum_data = DrumData(onsets=onsets, tempo=120.0)
    grid = build_drum_grid(drum_data)
    assert grid.num_columns > 0
    # Kick at time 0 → column 0
    assert grid.grid["BD"][0] == "o"
    # At 120 BPM, beat_duration=0.5s, cell_duration=0.5/4=0.125s
    # Snare at time 0.5s → col = round(0.5 / 0.125) = 4
    assert grid.grid["S "][4] == "o"


# ---------------------------------------------------------------------------
# ASCII rendering (smoke tests)
# ---------------------------------------------------------------------------

def _make_guitar_tab():
    notes = [Note(pitch=64, start=0.0, end=0.4), Note(pitch=60, start=0.5, end=0.9)]
    midi = MidiData(notes=notes, tempo=120.0)
    return guitar_assign_frets(midi)

def _make_bass_tab():
    notes = [Note(pitch=28, start=0.0, end=0.4), Note(pitch=33, start=0.5, end=0.9)]
    midi = MidiData(notes=notes, tempo=120.0)
    return bass_assign_frets(midi)

def test_render_guitar_tab_contains_strings():
    tab = _make_guitar_tab()
    output = render_guitar_tab(tab)
    assert "e|" in output or "E|" in output

def test_render_bass_tab_contains_strings():
    tab = _make_bass_tab()
    output = render_bass_tab(tab)
    assert "G|" in output or "E|" in output

def test_render_drum_tab():
    onsets = [DrumOnset(time=0.0, part="kick"), DrumOnset(time=0.5, part="snare")]
    drum_data = DrumData(onsets=onsets, tempo=120.0)
    grid = build_drum_grid(drum_data)
    output = render_drum_tab(grid)
    assert "BD" in output
    assert "S " in output

def test_render_vocal_staff():
    from tabs_gen.utils.midi_utils import MidiData, Note
    notes = [Note(pitch=69, start=0.0, end=1.0), Note(pitch=71, start=1.0, end=1.5)]
    midi = MidiData(notes=notes, tempo=120.0)
    from tabs_gen.stages.notation.vocals import build_vocal_staff
    staff = build_vocal_staff(midi)
    output = render_vocal_staff(staff)
    assert "A4" in output

def test_render_full_tab_smoke():
    guitar_tab = _make_guitar_tab()
    output = render_full_tab(guitar_tab=guitar_tab, title="Test Song")
    assert "Test Song" in output
    assert "GUITAR" in output
