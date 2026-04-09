# tabs-gen

Generate instrument tabs from any audio file using ML-based source separation and transcription.

Given an MP3 or WAV, `tabs-gen` outputs:
- Guitar tabs (ASCII + Guitar Pro)
- Bass tabs (ASCII + Guitar Pro)
- Drum tabs (ASCII)
- Vocal melody notation (ASCII)

Output quality is **draft-quality** — suitable as a starting point for manual refinement, not a finished product. See [Quality Expectations](#quality-expectations) for details.

## How it works

```
Audio file
    │
    ▼
Stage 1 — Source separation (Demucs htdemucs_6s)
    │        Splits into: vocals / guitar / bass / drums / piano / other
    ▼
Stage 2 — Transcription per stem
    │        Guitar/Bass → basic-pitch (Spotify)
    │        Vocals      → CREPE pitch tracker
    │        Drums       → ADTLib onset detection
    ▼
Stage 3 — Notation mapping
    │        Guitar/Bass → dynamic programming fret/string solver
    │        Drums       → General MIDI drum map → ASCII grid
    │        Vocals      → pitch quantisation → note staff
    ▼
Stage 4 — Output
             ASCII text tabs (.txt)
             Guitar Pro 5 file (.gp5)
```

## Requirements

- Python 3.10+
- `ffmpeg` system binary (for MP3 decoding)
- Apple Silicon Mac recommended (MPS acceleration); NVIDIA GPU (CUDA) or CPU also supported

## Installation

```bash
# 1. Clone the repo
git clone <repo-url>
cd tabs-gen

# 2. Create and activate a virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install the package
pip install -e .

# 4. Install ML dependencies
#    Source separation (PyTorch)
pip install demucs

#    Transcription (TensorFlow — guitar, bass, vocals)
pip install basic-pitch crepe

#    Drum transcription
pip install madmom ADTLib

#    Guitar Pro output
pip install pyguitarpro
```

> **Dependency note**: `basic-pitch` and `demucs` use TensorFlow and PyTorch respectively.
> Both coexist on Apple Silicon. If you hit conflicts on other platforms, use a
> [conda environment](https://docs.conda.io/en/latest/) instead of venv.

### ffmpeg

```bash
brew install ffmpeg        # macOS
sudo apt install ffmpeg    # Ubuntu/Debian
```

## Usage

```bash
# Full pipeline — all instruments, ASCII + GP5 output
tabs-gen song.mp3

# Specify output directory
tabs-gen song.mp3 --output ./my-tabs/

# Guitar and bass only
tabs-gen song.mp3 --instrument guitar --instrument bass

# ASCII text only (skip GP5)
tabs-gen song.mp3 --format ascii

# CPU mode (slower, no GPU required)
tabs-gen song.mp3 --device cpu

# Higher quality separation (slower — good for final output)
tabs-gen song.mp3 --shifts 10
```

### All options

```
Usage: tabs-gen [OPTIONS] AUDIO_FILE

  Generate instrument tabs from an audio file.

Options:
  -o, --output PATH               Output directory  [default: ./output]
  -f, --format [ascii|gp5]        Output format(s), repeatable  [default: ascii, gp5]
  -i, --instrument [guitar|bass|drums|vocals]
                                  Instruments to include, repeatable
  --model TEXT                    Demucs model: htdemucs, htdemucs_6s, htdemucs_ft
                                  [default: htdemucs_6s]
  --device TEXT                   Torch device: mps, cuda, cpu  [default: mps]
  --shifts INTEGER                Test-time shifts (1=fast, 10=best)  [default: 1]
  --onset-threshold FLOAT         basic-pitch onset threshold 0–1  [default: 0.5]
  --frame-threshold FLOAT         basic-pitch frame threshold 0–1  [default: 0.3]
  --crepe-model [tiny|small|medium|large|full]
                                  CREPE model for vocals  [default: medium]
  --title TEXT                    Song title in output files
  -v, --verbose                   Enable debug logging
  --help                          Show this message and exit
```

## Output

For an input file `song.mp3`, the output directory contains:

```
output/
├── song.txt          # ASCII tabs for all instruments
├── song.gp5          # Guitar Pro 5 file (guitar + bass + drums)
└── stems/
    ├── vocals.wav
    ├── guitar.wav
    ├── bass.wav
    ├── drums.wav
    ├── piano.wav
    └── other.wav
```

### ASCII tab example

```
============================================================
  Song Title
============================================================

[ GUITAR ]
 e|--0--------5--3--------0----|
 B|--0--------5--3--------0----|
 G|--1--------5--4--------1----|
 D|--2--------7--5--------2----|
 A|--2--------7--5--------2----|
 E|--0--------5--3--------0----|

[ BASS ]
 G|--------------------------------|
 D|--------------------------------|
 A|--0--------3--2--------0--------|
 E|--------------------------------|

[ DRUMS ]
HH|x-x-x-x-x-x-x-x-|
S |----o-------o----|
BD|o-----------o----|

[ VOCALS ]
Melody: A4(1.00) G4(0.50) F#4(0.50) E4(2.00) D4(1.00) ...
```

The `.gp5` file can be opened in:
- [Guitar Pro](https://www.guitar-pro.com/) (paid)
- [TuxGuitar](https://sourceforge.net/projects/tuxguitar/) (free)
- MuseScore (with the Guitar Pro import plugin)

## Quality Expectations

| Track | Typical accuracy | Notes |
|-------|-----------------|-------|
| Bass tabs | ~75% notes, ~85% rhythm | Best result; usable starting point |
| Guitar (single-note lines) | ~65% notes, ~55% fret positions | Rhythm ~70% |
| Guitar (chords/rhythm) | ~40–50% chord accuracy | Needs significant correction |
| Drums (kick/snare/hi-hat) | ~70% onset accuracy | Fills degrade to ~50% |
| Vocals (melody) | ~80% pitch, ~70% rhythm | Clean vocals only |

Results are best on **clean, professionally mixed rock/pop**. Heavy distortion, dense mixes, or complex polyrhythms reduce accuracy by ~15–20%.

## Development

```bash
# Run tests (no ML deps required)
source .venv/bin/activate
pip install pytest
pytest tests/ -v
```

Tests cover the notation and rendering stages (guitar/bass fret assignment, drum grid quantisation, ASCII rendering) without requiring any ML dependencies to be installed.

## Project structure

```
tabs_gen/
├── cli.py                    # Click CLI entry point
├── pipeline.py               # Orchestrates all 4 stages
├── stages/
│   ├── separation.py         # Stage 1: Demucs wrapper
│   ├── transcription.py      # Stage 2: basic-pitch, CREPE, ADTLib
│   ├── notation/
│   │   ├── guitar.py         # DP fret/string solver
│   │   ├── bass.py           # Bass tab assignment
│   │   ├── drums.py          # Drum MIDI map → grid
│   │   └── vocals.py         # Pitch contour → note staff
│   └── output/
│       ├── ascii_tab.py      # ASCII tab renderer
│       └── gp5.py            # PyGuitarPro GP5 writer
└── utils/
    ├── audio.py              # Audio I/O helpers
    ├── midi_utils.py         # MIDI parsing, quantisation
    └── rhythm.py             # BPM detection, beat grid
```

## Limitations and known issues

- **Chords**: Basic Pitch's polyphonic transcription of a rhythm guitar stem produces a rough skeleton; expect significant correction needed for chord songs.
- **Drum fills and toms**: ADTLib reliably detects kick/snare/hi-hat. Toms and cymbals are best-effort and may be missing or misidentified.
- **Guitar position ambiguity**: The DP solver minimises hand movement and prefers open/low-fret positions, but cannot know the original player's intent (e.g., preference for a specific tonal position).
- **Vocal slides and vibrato**: Pitch bends are discretised to the nearest semitone. Portamento and vibrato are not currently encoded in the output.
- **Piano stem**: Separated by Demucs but not yet transcribed or included in output (v2 roadmap).
- **GP5 drum track**: Percussion track structure in PyGuitarPro has known edge cases; verify playback in TuxGuitar after generation.
