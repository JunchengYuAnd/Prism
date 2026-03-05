"""
Prism — Dataset Generator
Renders audio samples from Serum via DawDreamer with randomized synth parameters.

Usage:
    python -m dataset.generate                        # full generation
    python -m dataset.generate --dry_run              # 10 samples only
    python -m dataset.generate --workers 4            # parallel rendering
    python -m dataset.generate --dry_run --workers 2  # combined
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import dawdreamer as daw
import numpy as np
import soundfile as sf
from tqdm import tqdm

from dataset import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in (config.AUDIO_DIR, config.METADATA_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _amplitude_db(audio: np.ndarray) -> float:
    """Return the peak amplitude of an audio array in dB."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return -np.inf
    return 20.0 * np.log10(peak)


def _load_existing_ids() -> set[str]:
    """Read metadata.csv and return the set of already-generated sample IDs."""
    if not config.METADATA_CSV.exists():
        return set()
    ids: set[str] = set()
    with open(config.METADATA_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.add(row["sample_id"])
    return ids


def _init_csv(param_names: list[str]) -> None:
    """Write the CSV header if the file doesn't exist yet."""
    if config.METADATA_CSV.exists():
        return
    with open(config.METADATA_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "file_path"] + param_names)


def _append_csv(sample_id: str, file_path: str, param_values: list[float]) -> None:
    """Append one row to metadata.csv."""
    with open(config.METADATA_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([sample_id, file_path] + [f"{v:.6f}" for v in param_values])


# ─── Core rendering ──────────────────────────────────────────────────────────

def _dump_parameter_list(plugin: Any) -> None:
    """Print and save the full Serum parameter list for user inspection."""
    desc = plugin.get_plugin_parameters_description()
    logger.info("Serum has %d parameters. Saving list to %s", len(desc), config.PARAMETER_LIST_FILE)
    config.METADATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.PARAMETER_LIST_FILE, "w") as f:
        for i, param in enumerate(desc):
            line = f"[{i:>4d}] {param['name']:<40s}  default={param['defaultValue']:.4f}  min={param['minValue']:.4f}  max={param['maxValue']:.4f}"
            f.write(line + "\n")
    logger.info(
        "Parameter list saved. Inspect %s, then fill SYNTH_PARAMETERS in config.py.",
        config.PARAMETER_LIST_FILE,
    )


def _create_engine_and_plugin() -> tuple[Any, Any]:
    """Instantiate a DawDreamer engine and load Serum."""
    engine = daw.RenderEngine(config.SAMPLE_RATE, 512)
    plugin = engine.make_plugin_processor("Serum", config.SERUM_VST_PATH)
    if config.SERUM_BASE_PRESET:
        plugin.load_preset(config.SERUM_BASE_PRESET)
        logger.info("Loaded base preset: %s", config.SERUM_BASE_PRESET)
    return engine, plugin


def _render_one(
    engine: Any,
    plugin: Any,
    param_indices: dict[str, int],
    param_values: dict[str, float],
) -> np.ndarray | None:
    """Set parameters, trigger a note, render, and return normalised mono audio (or None if silent)."""
    # Set randomised parameters
    for name, idx in param_indices.items():
        plugin.set_parameter(idx, param_values[name])

    # MIDI: note on at t=0, note off slightly before end
    plugin.add_midi_note(config.MIDI_NOTE, config.MIDI_VELOCITY, 0.0, config.AUDIO_DURATION - 0.05)

    graph = [
        (plugin, []),
    ]
    engine.load_graph(graph)
    engine.render(config.AUDIO_DURATION)
    audio = engine.get_audio()  # shape: (channels, samples)

    # Mix to mono
    if audio.ndim == 2:
        audio = audio.mean(axis=0)

    # Silence check
    if _amplitude_db(audio) < config.SILENCE_THRESHOLD_DB:
        return None

    # Peak-normalize to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak

    return audio


# ─── Single-threaded generation loop ─────────────────────────────────────────

def _generate_loop(total: int) -> None:
    """Main generation loop (single process)."""
    param_names = sorted(config.SYNTH_PARAMETERS.keys())
    param_indices = config.SYNTH_PARAMETERS

    _init_csv(param_names)
    existing_ids = _load_existing_ids()
    already_done = len(existing_ids)
    if already_done > 0:
        logger.info("Resuming: %d samples already generated, %d remaining.", already_done, total - already_done)

    engine, plugin = _create_engine_and_plugin()

    # Dump parameter list on first run
    if not config.PARAMETER_LIST_FILE.exists():
        _dump_parameter_list(plugin)

    if not param_indices:
        logger.warning(
            "SYNTH_PARAMETERS is empty. Only the parameter list was saved. "
            "Fill in config.py and re-run."
        )
        return

    silent_count = 0
    pbar = tqdm(total=total, initial=already_done, desc="Generating", unit="sample")

    generated = already_done
    while generated < total:
        sample_id = uuid.uuid4().hex[:12]
        if sample_id in existing_ids:
            continue

        # Random parameter values in [0, 1]
        param_values = {name: float(np.random.uniform(0, 1)) for name in param_names}

        audio = _render_one(engine, plugin, param_indices, param_values)

        if audio is None:
            silent_count += 1
            logger.debug("Silent sample skipped (total silent: %d)", silent_count)
            continue

        # Save .wav
        out_path = config.AUDIO_DIR / f"{sample_id}.wav"
        sf.write(str(out_path), audio, config.SAMPLE_RATE)

        # Append metadata
        _append_csv(sample_id, str(out_path.relative_to(config.PROJECT_ROOT)), [param_values[n] for n in param_names])
        existing_ids.add(sample_id)
        generated += 1
        pbar.update(1)

    pbar.close()
    logger.info("Done. Generated %d samples (%d silent skipped).", total, silent_count)


# ─── Parallel wrapper (multiprocessing) ──────────────────────────────────────

def _worker_fn(args: tuple[int, int]) -> tuple[int, int]:
    """Worker for parallel generation. Returns (generated, silent_skipped)."""
    worker_id, num_samples = args
    np.random.seed(os.getpid() + worker_id)

    param_names = sorted(config.SYNTH_PARAMETERS.keys())
    param_indices = config.SYNTH_PARAMETERS
    engine, plugin = _create_engine_and_plugin()

    generated = 0
    silent = 0
    for _ in range(num_samples):
        sample_id = uuid.uuid4().hex[:12]
        param_values = {name: float(np.random.uniform(0, 1)) for name in param_names}
        audio = _render_one(engine, plugin, param_indices, param_values)
        if audio is None:
            silent += 1
            continue
        out_path = config.AUDIO_DIR / f"{sample_id}.wav"
        sf.write(str(out_path), audio, config.SAMPLE_RATE)
        _append_csv(sample_id, str(out_path.relative_to(config.PROJECT_ROOT)), [param_values[n] for n in param_names])
        generated += 1

    return generated, silent


def _generate_parallel(total: int, workers: int) -> None:
    """Parallel generation using multiprocessing."""
    from multiprocessing import Pool

    param_names = sorted(config.SYNTH_PARAMETERS.keys())
    _init_csv(param_names)

    existing_ids = _load_existing_ids()
    already_done = len(existing_ids)
    remaining = total - already_done
    if remaining <= 0:
        logger.info("All %d samples already generated.", total)
        return

    if not config.SYNTH_PARAMETERS:
        # Still dump params on first run
        engine, plugin = _create_engine_and_plugin()
        if not config.PARAMETER_LIST_FILE.exists():
            _dump_parameter_list(plugin)
        logger.warning("SYNTH_PARAMETERS is empty. Fill config.py and re-run.")
        return

    logger.info("Spawning %d workers for %d remaining samples...", workers, remaining)
    chunk = remaining // workers
    tasks = [(i, chunk) for i in range(workers)]
    # Distribute remainder
    leftover = remaining - chunk * workers
    if leftover > 0:
        tasks[-1] = (workers - 1, chunk + leftover)

    with Pool(processes=workers) as pool:
        results = pool.map(_worker_fn, tasks)

    total_gen = sum(r[0] for r in results)
    total_silent = sum(r[1] for r in results)
    logger.info("Done. Generated %d samples (%d silent skipped).", total_gen, total_silent)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Prism dataset generator")
    parser.add_argument("--dry_run", action="store_true", help="Generate only 10 samples for testing")
    parser.add_argument("--workers", type=int, default=config.PARALLEL_WORKERS, help="Number of parallel workers")
    args = parser.parse_args()

    # Validate VST path
    if not config.SERUM_VST_PATH:
        print(
            "\n[ERROR] SERUM_VST_PATH is not set.\n"
            "  1. Open dataset/config.py\n"
            "  2. Set SERUM_VST_PATH to the full path of your Serum plugin\n"
            "     macOS example: '/Library/Audio/Plug-Ins/Components/Serum.component'\n"
            "     Windows example: r'C:\\Program Files\\VSTPlugins\\Serum_x64.dll'\n"
            "  3. Optionally set SERUM_BASE_PRESET to a .fxp preset file\n"
        )
        sys.exit(1)

    total = 10 if args.dry_run else config.TOTAL_SAMPLES
    _ensure_dirs()

    if args.workers > 1:
        _generate_parallel(total, args.workers)
    else:
        _generate_loop(total)


if __name__ == "__main__":
    main()
