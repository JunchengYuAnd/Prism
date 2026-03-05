"""
Prism — Feature Extraction
Extracts Linear Spectrogram and Mel Spectrogram from generated audio samples.

Usage:
    python -m dataset.extract_features
    python -m dataset.extract_features --workers 4
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
from tqdm import tqdm

from dataset import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Feature computation ─────────────────────────────────────────────────────

def compute_linear_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute log-amplitude linear spectrogram.
    Returns shape [1, freq_bins, time_frames] — e.g. [1, 1025, 173].
    """
    S = np.abs(
        librosa.stft(
            audio,
            n_fft=config.STFT_FFT_SIZE,
            hop_length=config.STFT_HOP_LENGTH,
            win_length=config.STFT_WINDOW_SIZE,
            window="hann",
        )
    )
    S_log = np.log(S + 1e-6)
    return S_log[np.newaxis, :, :]  # [1, F, T]


def compute_mel_spectrogram(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute log-amplitude Mel spectrogram.
    Returns shape [1, mel_bins, time_frames] — e.g. [1, 64, 173].
    """
    S = np.abs(
        librosa.stft(
            audio,
            n_fft=config.STFT_FFT_SIZE,
            hop_length=config.STFT_HOP_LENGTH,
            win_length=config.STFT_WINDOW_SIZE,
            window="hann",
        )
    )
    mel_basis = librosa.filters.mel(sr=sr, n_fft=config.STFT_FFT_SIZE, n_mels=config.MEL_BINS)
    S_mel = mel_basis @ S
    S_mel_log = np.log(S_mel + 1e-6)
    return S_mel_log[np.newaxis, :, :]  # [1, M, T]


# ─── Single-file extraction ──────────────────────────────────────────────────

def extract_one(audio_path_str: str) -> tuple[bool, str]:
    """
    Extract both spectrograms for a single audio file.
    Returns (success, message).
    """
    audio_path = config.PROJECT_ROOT / audio_path_str
    sample_id = Path(audio_path_str).stem

    linear_out = config.LINEAR_SPEC_DIR / f"{sample_id}.npy"
    mel_out = config.MEL_SPEC_DIR / f"{sample_id}.npy"

    # Skip if both already exist (resumable)
    if linear_out.exists() and mel_out.exists():
        return True, "skipped"

    if not audio_path.exists():
        return False, f"audio file not found: {audio_path}"

    try:
        audio, sr = librosa.load(str(audio_path), sr=config.SAMPLE_RATE, mono=True)
    except Exception as e:
        return False, f"failed to load {audio_path}: {e}"

    # Compute and save
    if not linear_out.exists():
        linear_spec = compute_linear_spectrogram(audio, sr)
        np.save(str(linear_out), linear_spec)

    if not mel_out.exists():
        mel_spec = compute_mel_spectrogram(audio, sr)
        np.save(str(mel_out), mel_spec)

    return True, "ok"


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prism feature extraction")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    args = parser.parse_args()

    # Read metadata to get file list
    if not config.METADATA_CSV.exists():
        print(
            "[ERROR] metadata.csv not found. Run generate.py first.\n"
            f"  Expected at: {config.METADATA_CSV}"
        )
        sys.exit(1)

    audio_paths: list[str] = []
    with open(config.METADATA_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            audio_paths.append(row["file_path"])

    if not audio_paths:
        print("[ERROR] metadata.csv is empty — no samples to process.")
        sys.exit(1)

    logger.info("Found %d samples in metadata.csv", len(audio_paths))

    # Ensure output dirs
    config.LINEAR_SPEC_DIR.mkdir(parents=True, exist_ok=True)
    config.MEL_SPEC_DIR.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []

    if args.workers > 1:
        with Pool(processes=args.workers) as pool:
            results = list(tqdm(
                pool.imap(extract_one, audio_paths),
                total=len(audio_paths),
                desc="Extracting features",
                unit="file",
            ))
    else:
        results = []
        for path in tqdm(audio_paths, desc="Extracting features", unit="file"):
            results.append(extract_one(path))

    # Summarise
    ok_count = sum(1 for success, _ in results if success)
    skip_count = sum(1 for success, msg in results if success and msg == "skipped")
    fail_count = sum(1 for success, _ in results if not success)

    for success, msg in results:
        if not success:
            errors.append(msg)

    logger.info(
        "Extraction complete: %d succeeded (%d skipped, already existed), %d failed.",
        ok_count, skip_count, fail_count,
    )

    if errors:
        logger.warning("Errors:")
        for e in errors[:20]:
            logger.warning("  %s", e)
        if len(errors) > 20:
            logger.warning("  ... and %d more errors.", len(errors) - 20)


if __name__ == "__main__":
    main()
