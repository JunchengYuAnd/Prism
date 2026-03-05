"""
Prism — Dataset Verification
Checks integrity of generated audio, feature files, and metadata.

Usage:
    python -m dataset.verify
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import librosa
import numpy as np

from dataset import config


def _load_metadata() -> list[dict[str, str]]:
    """Load metadata.csv and return list of row dicts."""
    if not config.METADATA_CSV.exists():
        print(f"[ERROR] metadata.csv not found at {config.METADATA_CSV}")
        sys.exit(1)
    with open(config.METADATA_CSV, "r") as f:
        return list(csv.DictReader(f))


def _amplitude_db(audio: np.ndarray) -> float:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return -np.inf
    return 20.0 * np.log10(peak)


def main() -> None:
    rows = _load_metadata()
    total = len(rows)

    if total == 0:
        print("[ERROR] metadata.csv is empty.")
        sys.exit(1)

    # Detect parameter columns (everything except sample_id and file_path)
    param_names = [k for k in rows[0].keys() if k not in ("sample_id", "file_path")]

    # Accumulators
    missing_audio: list[str] = []
    missing_linear: list[str] = []
    missing_mel: list[str] = []
    silent_samples: list[str] = []
    param_values: dict[str, list[float]] = {name: [] for name in param_names}

    print(f"Verifying {total} samples...\n")

    for row in rows:
        sid = row["sample_id"]
        audio_path = config.PROJECT_ROOT / row["file_path"]
        linear_path = config.LINEAR_SPEC_DIR / f"{sid}.npy"
        mel_path = config.MEL_SPEC_DIR / f"{sid}.npy"

        # Check audio existence
        if not audio_path.exists():
            missing_audio.append(sid)
        else:
            # Check silence
            try:
                audio, _ = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
                if _amplitude_db(audio) < config.SILENCE_THRESHOLD_DB:
                    silent_samples.append(sid)
            except Exception:
                missing_audio.append(sid)  # treat corrupt files as missing

        # Check feature files
        if not linear_path.exists():
            missing_linear.append(sid)
        if not mel_path.exists():
            missing_mel.append(sid)

        # Collect parameter values
        for name in param_names:
            try:
                param_values[name].append(float(row[name]))
            except (ValueError, KeyError):
                pass

    # ─── Report ───────────────────────────────────────────────────────────

    print("=" * 60)
    print("  PRISM DATASET VERIFICATION REPORT")
    print("=" * 60)

    print(f"\n  Total samples in metadata:  {total}")
    print(f"  Missing audio files:        {len(missing_audio)}")
    print(f"  Missing linear spec (.npy): {len(missing_linear)}")
    print(f"  Missing mel spec (.npy):    {len(missing_mel)}")
    print(f"  Silent samples detected:    {len(silent_samples)}")

    # Parameter distribution
    if param_names:
        print(f"\n  Parameter distributions ({len(param_names)} parameters):")
        print(f"  {'Name':<30s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Std':>8s}")
        print("  " + "-" * 56)
        for name in param_names:
            vals = param_values[name]
            if vals:
                arr = np.array(vals)
                print(
                    f"  {name:<30s} {arr.min():8.4f} {arr.max():8.4f} "
                    f"{arr.mean():8.4f} {arr.std():8.4f}"
                )
            else:
                print(f"  {name:<30s}  (no data)")

    # Anomalies
    anomalies: list[str] = []
    if missing_audio:
        anomalies.append(f"{len(missing_audio)} missing audio files")
    if missing_linear:
        anomalies.append(f"{len(missing_linear)} missing linear spectrogram files")
    if missing_mel:
        anomalies.append(f"{len(missing_mel)} missing mel spectrogram files")
    if silent_samples:
        anomalies.append(f"{len(silent_samples)} silent samples (should have been filtered)")

    if anomalies:
        print("\n  Anomalies found:")
        for a in anomalies:
            print(f"    - {a}")
    else:
        print("\n  No anomalies found. Dataset is clean.")

    print()


if __name__ == "__main__":
    main()
