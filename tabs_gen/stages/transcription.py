"""Stage 2: Audio-to-MIDI transcription per stem.

Instrument routing:
  guitar  → basic-pitch (polyphonic AMT)
  bass    → basic-pitch (semi-monophonic, low-freq range restricted)
  vocals  → CREPE pitch tracker + contour segmentation
  drums   → ADTLib (onset detection + drum-type classifier)

Each function accepts a path to a stem WAV and returns a MidiData object
(or a list of drum onset events for the drums stem).
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tabs_gen.utils.midi_utils import MidiData, Note, load_midi, pitch_contour_to_notes
from tabs_gen.utils.rhythm import detect_tempo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guitar / Bass via basic-pitch
# ---------------------------------------------------------------------------

def transcribe_pitched(
    stem_path: str | Path,
    min_freq: float | None = None,
    max_freq: float | None = None,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    min_note_length_ms: int = 58,
    instrument_name: str = "guitar",
) -> MidiData:
    """Transcribe a pitched instrument stem using Spotify's basic-pitch.

    Args:
        stem_path: path to a separated WAV stem.
        min_freq: low-pass frequency cut for pitch detection (Hz).
        max_freq: high-pass frequency cut for pitch detection (Hz).
        onset_threshold: basic-pitch onset detection sensitivity (0–1).
        frame_threshold: basic-pitch frame activation threshold (0–1).
        min_note_length_ms: minimum note duration in ms.
        instrument_name: label stored in the returned MidiData.

    Returns:
        MidiData containing detected notes with tempo estimate.
    """
    try:
        from basic_pitch.inference import predict  # type: ignore
        from basic_pitch import ICASSP_2022_MODEL_PATH  # type: ignore
    except ImportError as e:
        raise ImportError(
            "basic-pitch is required for pitched transcription. "
            "Install it with: pip install basic-pitch"
        ) from e

    stem_path = Path(stem_path)
    logger.info("  basic-pitch transcribing '%s'…", stem_path.name)

    # Prefer ONNX model over TF saved model (avoids TF version compatibility issues)
    import pathlib
    _tf_model_path = pathlib.Path(ICASSP_2022_MODEL_PATH)
    _onnx_path = _tf_model_path.with_suffix(".onnx")
    _model_path = str(_onnx_path) if _onnx_path.exists() else ICASSP_2022_MODEL_PATH

    model_output, midi_data, note_events = predict(
        stem_path,
        _model_path,
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=min_note_length_ms,
        minimum_frequency=min_freq,
        maximum_frequency=max_freq,
    )

    # midi_data is a pretty_midi.PrettyMIDI object
    notes: list[Note] = []
    for inst in midi_data.instruments:
        for n in inst.notes:
            notes.append(Note(pitch=n.pitch, start=n.start, end=n.end, velocity=n.velocity))

    notes.sort(key=lambda n: n.start)

    # Estimate tempo from the audio for quantization
    import soundfile as sf
    audio, sr = sf.read(str(stem_path), dtype="float32", always_2d=False)
    tempo = detect_tempo(audio, sr)

    return MidiData(notes=notes, tempo=tempo, instrument_name=instrument_name)


def transcribe_guitar(stem_path: str | Path, **kwargs) -> MidiData:
    """Transcribe a guitar stem (full frequency range, polyphonic)."""
    return transcribe_pitched(
        stem_path,
        min_freq=80.0,    # below low E string on guitar (82 Hz)
        max_freq=1320.0,  # high E, 24th fret ≈ 1318 Hz
        instrument_name="guitar",
        **kwargs,
    )


def transcribe_bass(stem_path: str | Path, **kwargs) -> MidiData:
    """Transcribe a bass guitar stem (restricted low-frequency range)."""
    return transcribe_pitched(
        stem_path,
        min_freq=30.0,   # below open B string on 5-string bass
        max_freq=400.0,  # ~G4, well above highest bass note
        instrument_name="bass",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Vocals via CREPE
# ---------------------------------------------------------------------------

def transcribe_vocals(
    stem_path: str | Path,
    model_capacity: str = "medium",
    confidence_threshold: float = 0.5,
    viterbi: bool = True,
) -> MidiData:
    """Transcribe vocal melody using CREPE pitch estimation.

    Args:
        stem_path: path to separated vocals WAV.
        model_capacity: CREPE model size — "tiny"|"small"|"medium"|"large"|"full".
        confidence_threshold: frames with confidence below this are unvoiced.
        viterbi: use Viterbi decoding for smoother pitch contour.

    Returns:
        MidiData with one note per voiced segment.
    """
    try:
        import crepe  # type: ignore
    except ImportError as e:
        raise ImportError(
            "crepe is required for vocal transcription. "
            "Install it with: pip install crepe"
        ) from e

    import soundfile as sf

    stem_path = Path(stem_path)
    logger.info("  CREPE transcribing vocals from '%s'…", stem_path.name)

    audio, sr = sf.read(str(stem_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo → mono

    # CREPE expects mono audio at any sample rate; internally resamples to 16 kHz
    times, frequencies, confidence, _ = crepe.predict(
        audio,
        sr,
        model_capacity=model_capacity,
        viterbi=viterbi,
        step_size=10,  # 10ms hop
    )

    notes = pitch_contour_to_notes(
        times=times,
        frequencies=frequencies,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
    )

    tempo = detect_tempo(audio, sr)
    return MidiData(notes=notes, tempo=tempo, instrument_name="vocals")


# ---------------------------------------------------------------------------
# Drums via ADTLib
# ---------------------------------------------------------------------------

# General MIDI drum map (channel 10 note → drum part name)
GM_DRUM_MAP: dict[int, str] = {
    35: "kick",   36: "kick",
    38: "snare",  40: "snare",
    42: "hi_hat_closed",
    44: "hi_hat_pedal",
    46: "hi_hat_open",
    49: "crash",  57: "crash",
    51: "ride",   59: "ride",
    41: "tom_low",  43: "tom_low",
    45: "tom_mid",  47: "tom_mid",
    48: "tom_high", 50: "tom_high",
}


@dataclass
class DrumOnset:
    """A single drum hit event."""
    time: float          # onset in seconds
    part: str            # drum part name (kick, snare, hi_hat_closed, …)
    midi_note: int = 36  # GM drum note
    velocity: int = 80


@dataclass
class DrumData:
    """Drum transcription result."""
    onsets: list[DrumOnset] = field(default_factory=list)
    tempo: float = 120.0

    def sort(self) -> None:
        self.onsets.sort(key=lambda o: o.time)


def transcribe_drums(stem_path: str | Path) -> DrumData:
    """Transcribe drums using ADTLib (kick, snare, hi-hat).

    Falls back to a librosa onset-detection approach if ADTLib is not installed.

    Returns:
        DrumData with onset events for each detected drum component.
    """
    stem_path = Path(stem_path)
    logger.info("  Transcribing drums from '%s'…", stem_path.name)

    try:
        return _transcribe_drums_adtlib(stem_path)
    except ImportError:
        logger.warning("ADTLib not available; falling back to librosa onset detection")
        return _transcribe_drums_librosa(stem_path)


def _transcribe_drums_adtlib(stem_path: Path) -> DrumData:
    try:
        import ADTLib  # type: ignore  # noqa: N813
    except ImportError as e:
        raise ImportError(
            "ADTLib is required for drum transcription. "
            "Install it with: pip install ADTLib"
        ) from e

    import soundfile as sf

    audio, sr = sf.read(str(stem_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    tempo = detect_tempo(audio, sr)

    # ADTLib returns dict: {"BD": [...times], "SD": [...times], "HH": [...times]}
    with tempfile.TemporaryDirectory() as tmp:
        result = ADTLib.predict(str(stem_path), output_dir=tmp)

    part_map = {
        "BD": ("kick", 36),
        "SD": ("snare", 38),
        "HH": ("hi_hat_closed", 42),
    }

    onsets: list[DrumOnset] = []
    for adtlib_key, (part_name, midi_note) in part_map.items():
        for t in result.get(adtlib_key, []):
            onsets.append(DrumOnset(time=float(t), part=part_name, midi_note=midi_note))

    dd = DrumData(onsets=onsets, tempo=tempo)
    dd.sort()
    return dd


def _transcribe_drums_librosa(stem_path: Path) -> DrumData:
    """Fallback drum transcription using librosa onset detection.

    Detects all onsets and labels them based on spectral centroid:
    - Low centroid  → kick
    - Mid centroid  → snare
    - High centroid → hi-hat
    """
    import librosa
    import soundfile as sf

    audio, sr = sf.read(str(stem_path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    tempo = detect_tempo(audio, sr)

    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units="frames")
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)

    hop_length = 512
    spec = np.abs(librosa.stft(audio, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr)
    centroids = librosa.feature.spectral_centroid(S=spec, freq=freqs)[0]

    onsets: list[DrumOnset] = []
    for t in onset_times:
        frame = librosa.time_to_frames(t, sr=sr, hop_length=hop_length)
        frame = min(frame, len(centroids) - 1)
        centroid = centroids[frame]

        if centroid < 300:
            part, note = "kick", 36
        elif centroid < 2000:
            part, note = "snare", 38
        else:
            part, note = "hi_hat_closed", 42

        onsets.append(DrumOnset(time=float(t), part=part, midi_note=note))

    dd = DrumData(onsets=onsets, tempo=tempo)
    dd.sort()
    return dd
