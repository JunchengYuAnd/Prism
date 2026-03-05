"""
Prism Dataset Configuration
All configurable values for dataset generation and feature extraction.
"""

from pathlib import Path

# ─── VST & Preset ────────────────────────────────────────────────────────────

# Path to Serum VST plugin (.component / .vst3 / .dll)
# Set this before running generate.py
SERUM_VST_PATH: str = "/Library/Audio/Plug-Ins/Components/Serum.component"

# Path to the base preset (.fxp) that defines the fixed wavetable
SERUM_BASE_PRESET: str = ""

# ─── Output Directories ──────────────────────────────────────────────────────

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_ROOT: Path = PROJECT_ROOT / "data"

AUDIO_DIR: Path = DATA_ROOT / "audio"
LINEAR_SPEC_DIR: Path = DATA_ROOT / "linear_spec"
MEL_SPEC_DIR: Path = DATA_ROOT / "mel_spec"
METADATA_DIR: Path = DATA_ROOT / "metadata"
METADATA_CSV: Path = METADATA_DIR / "metadata.csv"
PARAMETER_LIST_FILE: Path = METADATA_DIR / "serum_parameters.txt"

# ─── Audio Rendering ─────────────────────────────────────────────────────────

SAMPLE_RATE: int = 44100
AUDIO_DURATION: float = 2.0          # seconds
MIDI_NOTE: int = 60                  # C4
MIDI_VELOCITY: int = 100
TOTAL_SAMPLES: int = 100
PARALLEL_WORKERS: int = 1            # DawDreamer is not always thread-safe; start with 1
SILENCE_THRESHOLD_DB: float = -60.0  # skip samples quieter than this

# ─── STFT / Spectrogram ──────────────────────────────────────────────────────

STFT_WINDOW_SIZE: int = 2048
STFT_HOP_LENGTH: int = 512
STFT_FFT_SIZE: int = 2048
MEL_BINS: int = 64

# ─── Synth Parameters ────────────────────────────────────────────────────────
# Maps human-readable name → DawDreamer parameter index.
# Leave empty initially; fill after inspecting the parameter list printed
# by generate.py on its first run.
#
# Example:
#   SYNTH_PARAMETERS = {
#       "osc_a_level": 5,
#       "osc_b_level": 12,
#       "filter_cutoff": 23,
#       "filter_resonance": 24,
#   }

SYNTH_PARAMETERS: dict[str, int] = {
    # Oscillator A
    "a_vol": 1,             # Osc A volume
    "a_pan": 2,             # Osc A pan
    "a_unison_detune": 7,   # Osc A unison detune
    "a_unison_blend": 8,    # Osc A unison blend
    "a_wt_pos": 11,         # Osc A wavetable position
    # Filter
    "fil_cutoff": 45,       # Filter cutoff frequency
    "fil_resonance": 46,    # Filter resonance
    "fil_drive": 47,        # Filter drive
    "fil_mix": 49,          # Filter mix (dry/wet)
    # Envelope 1 (amp envelope)
    "env1_attack": 35,      # Amp envelope attack
    "env1_decay": 37,       # Amp envelope decay
    "env1_sustain": 38,     # Amp envelope sustain
    "env1_release": 39,     # Amp envelope release
    # Master
    "master_vol": 0,        # Master volume
}
