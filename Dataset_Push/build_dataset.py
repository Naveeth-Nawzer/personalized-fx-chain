"""Build the cleaned source-audio dataset in Hugging Face ``audiofolder`` layout.

Input (read only)::

    <source_dir>/<song>/bass.wav, drums.wav, mixture.wav, ...

Output::

    <output_dir>/
        README.md                       dataset card
        data/<split>/<sample_id>.wav    e.g. data/train/1-Bass.wav
        data/<split>/metadata.csv       one row per file (file_name + metadata)
        metadata/samples.csv            all samples, all splits
        metadata/sources.csv            source_id -> song -> split
        metadata/skipped.csv            stems dropped by the cleaning checks
        metadata/dataset_summary.json

Usage::

    python build_dataset.py                 # uses config.yaml next to this file
    python build_dataset.py --overwrite     # rebuild an existing output folder
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import wave
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
SUMMARY_FILE = "dataset_summary.json"

SAMPLE_COLUMNS = (
    "file_name", "sample_id", "source_id", "source_category", "source_song", "split",
    "sample_rate", "num_channels", "bit_depth", "duration_s", "rms_dbfs", "peak_dbfs",
)
SOURCE_COLUMNS = ("source_id", "source_song", "split")
SKIPPED_COLUMNS = ("sample_id", "source_id", "source_category", "source_song", "reason")


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = path.parent
    for key, value in config["paths"].items():
        config["paths"][key] = (base / value).resolve()
    if not config["source_categories"]:
        raise ValueError("config: source_categories must list at least one category")
    if not math.isclose(sum(config["split"][s] for s in SPLITS), 1.0, abs_tol=1e-6):
        raise ValueError("config: split ratios must sum to 1")
    return config


def make_sample_id(source_id: int, category: str) -> str:
    return f"{source_id}-{category}"


def discover_songs(source_dir: Path) -> list[str]:
    """Song folders, sorted case-insensitively so source_id numbering is stable."""
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")
    songs = [p.name for p in source_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    return sorted(songs, key=lambda s: (s.casefold(), s))


def allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Largest-remainder rounding; every split with a positive ratio gets >= 1 item when possible."""
    raw = {s: total * ratios[s] for s in SPLITS}
    counts = {s: math.floor(raw[s]) for s in SPLITS}
    for s in sorted(SPLITS, key=lambda s: -(raw[s] - counts[s]))[: total - sum(counts.values())]:
        counts[s] += 1
    for s in SPLITS:
        if ratios[s] > 0 and counts[s] == 0 and total >= len(SPLITS):
            donor = max(SPLITS, key=counts.get)
            counts[donor] -= 1
            counts[s] += 1
    return counts


def assign_splits(source_ids: list[int], ratios: dict[str, float], seed: int) -> dict[int, str]:
    order = list(source_ids)
    random.Random(seed).shuffle(order)
    assignment, start = {}, 0
    for split, count in allocate_counts(len(order), ratios).items():
        for sid in order[start : start + count]:
            assignment[sid] = split
        start += count
    return assignment


def analyse_wav(path: Path) -> dict:
    """Header info plus RMS/peak level (16/24/32-bit PCM WAV)."""
    with wave.open(str(path), "rb") as w:
        rate, channels, width, frames = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(frames)
    if width == 3:  # 24-bit: widen to int32
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        samples = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int8).astype(np.int32) << 16))
        full_scale = 2**23
    else:
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if width == 1:
            samples -= 128
        full_scale = 2 ** (8 * width - 1)
    x = np.asarray(samples, dtype=np.float64) / full_scale
    rms = float(np.sqrt(np.mean(x**2))) if x.size else 0.0
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    to_db = lambda v: round(20 * math.log10(v), 2) if v > 0 else float("-inf")
    return {
        "sample_rate": rate,
        "num_channels": channels,
        "bit_depth": 8 * width,
        "num_frames": frames,
        "duration_s": round(frames / rate, 3),
        "rms_dbfs": to_db(rms),
        "peak_dbfs": to_db(peak),
    }


def find_stem(song_dir: Path, category: str) -> Path | None:
    for path in song_dir.iterdir():
        if path.is_file() and path.suffix.lower() == ".wav" and path.stem.lower() == category.lower():
            return path
    return None


def write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            sys.exit(f"Output folder is not empty: {output_dir}\nRe-run with --overwrite to rebuild it.")
        if not (output_dir / "metadata" / SUMMARY_FILE).is_file():
            sys.exit(f"Refusing to delete {output_dir}: it was not created by this script.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def dataset_card(config: dict, summary: dict) -> str:
    categories = ", ".join(config["source_categories"])
    split_rows = "\n".join(
        f"| {s} | {summary['splits'][s]['num_sources']} | {summary['splits'][s]['num_samples']} |" for s in SPLITS
    )
    return f"""---
pretty_name: Inverse FX Source Audio (Cleaned)
license: other
license_name: musdb18
license_link: https://sigsep.github.io/datasets/musdb.html
task_categories:
- audio-classification
tags:
- audio
- music
- audio-effects
- musdb18
size_categories:
- n<1K
---

# Inverse FX Source Audio (Cleaned)

Clean (dry) source recordings used to generate the paired dry/wet data for the
**Inverse FX Chain Predictor**. Each sample is one stem of one song.

- Source categories: **{categories}**
- Format: WAV, {summary['audio']['sample_rate']} Hz, {summary['audio']['num_channels']} channel(s), {summary['audio']['bit_depth']}-bit PCM (copied unchanged from the source)
- Songs: {summary['num_sources']} | Samples: {summary['num_samples']}

## Naming

| field | description | example |
|---|---|---|
| `source_id` | song / source number | `1` |
| `source_category` | stem type | `Bass` |
| `sample_id` | `<source_id>-<source_category>` | `1-Bass` |

## Splits

Splits are made at **song level** (seed {config['seed']}, ratios
{config['split']['train']:.0%} / {config['split']['validation']:.0%} / {config['split']['test']:.0%}),
so every stem of a song is in the same split and no song leaks between splits.

| split | songs | samples |
|---|---|---|
{split_rows}

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{config['huggingface']['repo_id']}")
row = ds["train"][0]
row["sample_id"], row["source_category"], row["audio"]["sampling_rate"]
```

`metadata/sources.csv` maps each `source_id` to its original song name and split.

## License

Audio is taken from MUSDB18 and is provided for **non-commercial research use only**,
under the terms of the original dataset. Please cite MUSDB18 if you use this data.
"""


def build(config: dict, overwrite: bool) -> dict:
    source_dir, output_dir = config["paths"]["source_dir"], config["paths"]["output_dir"]
    categories = list(config["source_categories"])
    cleaning = config["cleaning"]

    songs = discover_songs(source_dir)
    if not songs:
        sys.exit(f"No song folders found in {source_dir}")
    source_ids = {song: i for i, song in enumerate(songs, start=1)}
    splits = assign_splits(list(source_ids.values()), config["split"], config["seed"])
    print(f"Found {len(songs)} songs; categories: {categories}")

    prepare_output_dir(output_dir, overwrite)

    samples, skipped, sample_rates = [], [], set()
    for song, sid in source_ids.items():
        split = splits[sid]
        for category in categories:
            sample_id = make_sample_id(sid, category)
            base = {"sample_id": sample_id, "source_id": sid, "source_category": category, "source_song": song}
            src = find_stem(source_dir / song, category)
            if src is None:
                skipped.append({**base, "reason": f"missing {category.lower()}.wav"})
                continue
            try:
                info = analyse_wav(src)
            except (wave.Error, EOFError, KeyError) as exc:
                skipped.append({**base, "reason": f"unreadable WAV: {exc}"})
                continue
            expected_rate = cleaning.get("expected_sample_rate")
            if expected_rate and info["sample_rate"] != expected_rate:
                skipped.append({**base, "reason": f"sample rate {info['sample_rate']} != {expected_rate}"})
                continue
            if info["num_frames"] == 0:
                skipped.append({**base, "reason": "empty file"})
                continue
            if cleaning.get("drop_silent", True) and info["rms_dbfs"] < cleaning["min_rms_dbfs"]:
                skipped.append({**base, "reason": f"silent (RMS {info['rms_dbfs']} dBFS)"})
                continue

            dest = output_dir / "data" / split / f"{sample_id}.wav"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            sample_rates.add(info["sample_rate"])
            samples.append({**base, **info, "split": split, "file_name": dest.name})
            print(f"  [{split:<10}] {sample_id:<14} <- {song}/{src.name}")

    if len(sample_rates) > 1:
        sys.exit(f"Inconsistent sample rates across files: {sorted(sample_rates)}")
    if not samples:
        sys.exit("No samples passed the cleaning checks.")

    # Per-split metadata.csv (file_name relative to the split folder) -> audiofolder loader.
    for split in SPLITS:
        write_csv(output_dir / "data" / split / "metadata.csv", [s for s in samples if s["split"] == split], SAMPLE_COLUMNS)
    meta_dir = output_dir / "metadata"
    write_csv(meta_dir / "samples.csv",
              [{**s, "file_name": f"data/{s['split']}/{s['file_name']}"} for s in samples], SAMPLE_COLUMNS)
    write_csv(meta_dir / "sources.csv",
              [{"source_id": sid, "source_song": song, "split": splits[sid]} for song, sid in source_ids.items()],
              SOURCE_COLUMNS)
    write_csv(meta_dir / "skipped.csv", skipped, SKIPPED_COLUMNS)

    first = samples[0]
    summary = {
        "dataset": config["huggingface"]["repo_id"],
        "source_categories": categories,
        "sample_id_format": "<source_id>-<source_category>",
        "num_sources": len(songs),
        "num_samples": len(samples),
        "num_skipped": len(skipped),
        "split_ratios": dict(config["split"]),
        "split_unit": "song (all categories of a song stay in one split)",
        "seed": config["seed"],
        "splits": {
            split: {
                "num_sources": sum(1 for sid in splits if splits[sid] == split),
                "num_samples": sum(1 for s in samples if s["split"] == split),
                "source_ids": sorted(sid for sid in splits if splits[sid] == split),
                "category_distribution": dict(Counter(s["source_category"] for s in samples if s["split"] == split)),
            }
            for split in SPLITS
        },
        "audio": {k: first[k] for k in ("sample_rate", "num_channels", "bit_depth")},
        "total_duration_s": round(sum(s["duration_s"] for s in samples), 1),
        "cleaning": dict(cleaning),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (meta_dir / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(dataset_card(config, summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--overwrite", action="store_true", help="rebuild an existing output folder")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    summary = build(config, args.overwrite)
    print(f"\nBuilt {summary['num_samples']} samples from {summary['num_sources']} songs "
          f"({summary['num_skipped']} skipped) -> {config['paths']['output_dir']}")
    for split in SPLITS:
        s = summary["splits"][split]
        print(f"  {split:<10} songs={s['num_sources']:<3} samples={s['num_samples']:<4} ids={s['source_ids']}")


if __name__ == "__main__":
    main()
