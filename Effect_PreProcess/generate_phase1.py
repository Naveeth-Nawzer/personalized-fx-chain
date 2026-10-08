"""Step 2 — generate the Phase 1 dry/wet effect-type dataset with Pedalboard.

For every source sample (e.g. ``1-Bass``) one dry clip is prepared and the 7
effect combinations are rendered in the fixed order EQ -> Compressor -> Reverb:

    1-Bass-EQ, 1-Bass-Compressor, 1-Bass-Reverb, 1-Bass-EQ-Compressor,
    1-Bass-EQ-Reverb, 1-Bass-Compressor-Reverb, 1-Bass-EQ-Compressor-Reverb

Output (``paths.output_dir``)::

    README.md                                   dataset card
    data/<split>/dry/<source_sample_id>.wav     e.g. data/train/dry/2-Bass.wav
    data/<split>/wet/<sample_id>.wav            e.g. data/train/wet/2-Bass-EQ-Compressor.wav
    metadata/<split>.csv                        one row per dry/wet pair
    metadata/sources.csv                        one row per dry clip
    metadata/skipped.csv, failures.csv, config_used.yaml, dataset_summary.json

The source dataset is only read, never modified.

Usage::

    .venv\\Scripts\\python generate_phase1.py
    .venv\\Scripts\\python generate_phase1.py --overwrite   # replace a previous run
    .venv\\Scripts\\python generate_phase1.py --limit 2     # quick test: first 2 sources only
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pedalboard
import yaml

from audio_utils import SourceRejected, amplitude_to_db, peak, prepare_dry_clip, write_audio
from common import DEFAULT_CONFIG, SUMMARY_FILE, load_config, read_csv, stable_hash, write_csv
from fx import LABEL_COLUMNS, LABEL_NAMES, PARAM_COLUMNS, ParameterSampler, effect_combinations, label_vector, render_chain

# Independent random streams derived from the single seed.
STREAM_CLIP, STREAM_PARAMS = 0, 1

SAMPLE_COLUMNS = (
    "sample_id", "source_sample_id", "source_id", "source_category", "source_song", "split",
    "dry_audio", "wet_audio", "effect_chain", "combination", "label",
    "eq_label", "compressor_label", "reverb_label",
    "eq_params", "compressor_params", "reverb_params",
    "parameters", "parameters_normalized", "effect_change_db", "low_effect_change",
    "sample_rate", "num_channels", "duration_s", "dry_peak_dbfs", "wet_peak_dbfs",
    "dsp_backend", "parameter_mode", "random_seed", "sample_seed",
)
SOURCE_COLUMNS = (
    "source_sample_id", "source_id", "source_category", "source_song", "split", "dry_audio", "source_file",
    "original_sample_rate", "original_channels", "original_duration_s", "clip_start_s", "clip_end_s",
    "active_fraction", "normalization", "input_loudness_lufs", "normalization_gain_db", "dry_peak_dbfs",
)
SKIPPED_COLUMNS = ("source_sample_id", "source_file", "split", "reason")
FAILURE_COLUMNS = ("sample_id", "source_sample_id", "effect_chain", "error")


def load_sources(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Source samples from the downloaded dataset, filtered and in a stable order."""
    source_dir = config["paths"]["source_dir"]
    index = source_dir / "metadata" / "samples.csv"
    if not index.is_file():
        sys.exit(f"Source dataset not found at {source_dir}. Run download_source.py first.")
    rows = read_csv(index)
    present = list(dict.fromkeys(r["source_category"] for r in rows))
    wanted = config["source"].get("categories") or present
    unknown = sorted(set(wanted) - set(present))
    if unknown:
        sys.exit(f"Categories {unknown} are not in the source dataset (available: {present})")
    bad_splits = {r["split"] for r in rows} - set(config["splits"])
    if bad_splits:
        sys.exit(f"Unexpected split names in source metadata: {sorted(bad_splits)}")
    rows = [r for r in rows if r["source_category"] in wanted]
    rows.sort(key=lambda r: (int(r["source_id"]), wanted.index(r["source_category"])))
    return rows


def prepare_output_dir(output_dir: Path, source_dir: Path, overwrite: bool) -> None:
    if output_dir.resolve().is_relative_to(source_dir.resolve()):
        sys.exit("output_dir must not be inside source_dir (the source dataset stays unchanged).")
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            sys.exit(f"{output_dir} already contains data. Re-run with --overwrite to replace it.")
        if not (output_dir / "metadata" / SUMMARY_FILE).is_file() and not (output_dir / "data").is_dir():
            sys.exit(f"Refusing to delete {output_dir}: it does not look like a generated dataset.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def generate(config: dict[str, Any], overwrite: bool, limit: int | None) -> dict[str, Any]:
    paths, gen = config["paths"], config["generation"]
    source_dir, output_dir = paths["source_dir"], paths["output_dir"]
    seed = int(gen["seed"])
    order = list(config["effect_order"])
    if sorted(order) != sorted(LABEL_NAMES):
        sys.exit(f"effect_order must contain exactly {list(LABEL_NAMES)}")
    combos = effect_combinations(order)
    sampler = ParameterSampler(config["parameters"], gen["parameter_mode"])
    backend = f"pedalboard {pedalboard.__version__}"

    sources = load_sources(config)
    if limit:
        sources = sources[:limit]
    prepare_output_dir(output_dir, source_dir, overwrite)
    print(f"{len(sources)} source samples x {len(combos)} combinations = {len(sources) * len(combos)} pairs "
          f"(seed {seed}, {gen['parameter_mode']} parameters, {backend})")

    records, source_rows, skipped, failures = [], [], [], []
    started = time.time()
    for k, src in enumerate(sources, start=1):
        src_sid, split = src["sample_id"], src["split"]
        clip_rng = np.random.default_rng(np.random.SeedSequence([seed, STREAM_CLIP, stable_hash(src_sid)]))
        try:
            clip = prepare_dry_clip(source_dir / src["file_name"], config["audio"], config["preprocessing"], clip_rng)
        except SourceRejected as exc:
            print(f"[{k}/{len(sources)}] SKIP {src_sid}: {exc}")
            skipped.append({"source_sample_id": src_sid, "source_file": src["file_name"], "split": split,
                            "reason": str(exc)})
            continue

        dry_rel = f"data/{split}/dry/{src_sid}.wav"
        write_audio(output_dir / dry_rel, clip.audio, clip.sample_rate, config["audio"]["output_subtype"])
        base = {"source_sample_id": src_sid, "source_id": int(src["source_id"]),
                "source_category": src["source_category"], "source_song": src["source_song"], "split": split}
        source_rows.append({**base, "dry_audio": dry_rel, "source_file": src["file_name"], **clip.info})

        for chain in combos:
            sample_id = f"{src_sid}-{'-'.join(chain)}"
            sample_seed = int(np.random.SeedSequence([seed, STREAM_PARAMS, stable_hash(sample_id)]).generate_state(1)[0])
            try:
                result = render_chain(clip.audio, clip.sample_rate, chain, order, sampler,
                                      np.random.default_rng(sample_seed),
                                      float(gen["min_effect_change_db"]), int(gen["max_retries"]))
            except Exception as exc:  # noqa: BLE001 — record and continue
                failures.append({"sample_id": sample_id, "source_sample_id": src_sid,
                                 "effect_chain": list(chain), "error": str(exc)})
                continue
            wet_rel = f"data/{split}/wet/{sample_id}.wav"
            write_audio(output_dir / wet_rel, result["wet"], clip.sample_rate, config["audio"]["output_subtype"])

            params = result["parameters"]
            label = label_vector(chain)
            records.append({
                "sample_id": sample_id, **base,
                "dry_audio": dry_rel, "wet_audio": wet_rel,
                "effect_chain": list(chain), "combination": "+".join(chain), "label": label,
                **{LABEL_COLUMNS[n]: v for n, v in zip(LABEL_NAMES, label)},
                **{PARAM_COLUMNS[n]: params.get(n) for n in LABEL_NAMES},
                "parameters": params,
                "parameters_normalized": {n: sampler.normalize(n, p) for n, p in params.items()},
                "effect_change_db": result["effect_change_db"],
                "low_effect_change": result["low_effect_change"],
                "sample_rate": clip.sample_rate, "num_channels": int(clip.audio.shape[0]),
                "duration_s": round(clip.audio.shape[1] / clip.sample_rate, 6),
                "dry_peak_dbfs": clip.info["dry_peak_dbfs"],
                "wet_peak_dbfs": round(amplitude_to_db(peak(result["wet"])), 3),
                "dsp_backend": backend, "parameter_mode": sampler.mode,
                "random_seed": seed, "sample_seed": sample_seed,
            })
        print(f"[{k}/{len(sources)}] {split:<10} {src_sid:<12} {src['source_song']}  ({time.time() - started:.0f}s)")

    if not records:
        sys.exit("No samples were generated.")
    summary = write_metadata(config, records, source_rows, skipped, failures, combos, backend)
    return summary


def write_metadata(config, records, source_rows, skipped, failures, combos, backend) -> dict[str, Any]:
    output_dir = config["paths"]["output_dir"]
    meta = output_dir / "metadata"
    splits = config["splits"]
    for split in splits:
        write_csv(meta / f"{split}.csv", [r for r in records if r["split"] == split], SAMPLE_COLUMNS)
    write_csv(meta / "sources.csv", source_rows, SOURCE_COLUMNS)
    write_csv(meta / "skipped.csv", skipped, SKIPPED_COLUMNS)
    write_csv(meta / "failures.csv", failures, FAILURE_COLUMNS)
    public = {k: v for k, v in config.items() if k != "paths"}
    (meta / "config_used.yaml").write_text(yaml.safe_dump(public, sort_keys=False), encoding="utf-8")

    songs = {s: sorted({r["source_id"] for r in source_rows if r["split"] == s}) for s in splits}
    total_songs = sum(len(v) for v in songs.values())
    summary = {
        "phase": "phase1_effect_type",
        "task": "multi-label effect-type classification from paired dry/wet audio",
        "source_dataset": config["source"]["repo_id"],
        "label_names": list(LABEL_NAMES),
        "processing_order": " -> ".join(config["effect_order"]),
        "effect_combinations": ["+".join(c) for c in combos],
        "sample_id_format": "<source_id>-<source_category>-<effect>[-<effect>...]",
        "num_sources": len(source_rows),
        "num_pairs": len(records),
        "num_expected_pairs": len(source_rows) * len(combos),
        "num_skipped_sources": len(skipped),
        "num_failed_pairs": len(failures),
        "splits": {
            s: {
                "num_songs": len(songs[s]),
                "song_fraction": round(len(songs[s]) / total_songs, 4) if total_songs else 0.0,
                "source_ids": songs[s],
                "num_sources": sum(1 for r in source_rows if r["split"] == s),
                "num_pairs": sum(1 for r in records if r["split"] == s),
            }
            for s in splits
        },
        "split_unit": "song (inherited from the source dataset; no song appears in two splits)",
        "category_distribution": dict(Counter(r["source_category"] for r in source_rows)),
        "combination_distribution": dict(Counter(r["combination"] for r in records)),
        "label_distribution": {n: sum(r[LABEL_COLUMNS[n]] for r in records) for n in LABEL_NAMES},
        "num_low_effect_change_pairs": sum(1 for r in records if r["low_effect_change"]),
        "audio": {k: config["audio"][k] for k in ("sample_rate", "channels", "duration", "output_subtype")},
        "preprocessing": dict(config["preprocessing"]),
        "parameter_mode": config["generation"]["parameter_mode"],
        "random_seed": config["generation"]["seed"],
        "dsp_backend": backend,
        "python": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (meta / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(dataset_card(config, summary), encoding="utf-8")
    return summary


def dataset_card(config: dict[str, Any], s: dict[str, Any]) -> str:
    split_rows = "\n".join(
        f"| {k} | {v['num_songs']} ({v['song_fraction']:.0%}) | {v['num_sources']} | {v['num_pairs']} |"
        for k, v in s["splits"].items()
    )
    combo_rows = "\n".join(
        f"| {i} | {c.replace('+', ' + ')} | `{label_vector(c.split('+'))}` |"
        for i, c in enumerate(s["effect_combinations"], start=1)
    )
    a = s["audio"]
    # The viewer / load_dataset read the Parquet shards written by export_parquet.py,
    # where dry_audio and wet_audio are typed as Audio.
    data_files = "\n".join(f"  - split: {k}\n    path: data/{k}-*.parquet" for k in s["splits"])
    return f"""---
pretty_name: Inverse FX Phase 1 — Effect-Type Dataset
license: other
license_name: musdb18
license_link: https://sigsep.github.io/datasets/musdb.html
task_categories:
- audio-classification
tags:
- audio
- music
- audio-effects
- pedalboard
- multi-label-classification
configs:
- config_name: default
  data_files:
{data_files}
---

# Inverse FX Phase 1 — Effect-Type Dataset

Paired **dry / wet** audio for multi-label effect-type prediction
(Inverse FX Chain Predictor, Phase 1). Generated from
[{s['source_dataset']}](https://huggingface.co/datasets/{s['source_dataset']}) with
[Pedalboard](https://github.com/spotify/pedalboard) ({s['dsp_backend']}).

- Effects: **EQ**, **Compressor**, **Reverb**, always applied in the fixed order **{s['processing_order']}**
- Label: binary vector `[EQ, Compressor, Reverb]` (effect presence only)
- Sources: {s['num_sources']} dry clips ({', '.join(f'{k}: {v}' for k, v in s['category_distribution'].items())})
- Pairs: {s['num_pairs']} ({len(s['effect_combinations'])} combinations per dry clip)
- Audio: {a['sample_rate']} Hz, {a['channels']} channel(s), {a['duration']} s clips, 32-bit float WAV
- Effect parameters: {s['parameter_mode']}, reproducible with seed {s['random_seed']}

## Effect combinations

| # | effect_chain | label |
|---|---|---|
{combo_rows}

## Splits (song level, inherited from the source dataset)

| split | songs | dry clips | dry/wet pairs |
|---|---|---|---|
{split_rows}

## Naming

| field | example |
|---|---|
| `source_sample_id` | `1-Bass` |
| `sample_id` | `1-Bass-EQ-Compressor` |
| `source_id` | `1` |
| `source_category` | `Bass` |
| `effect_chain` | `["EQ","Compressor"]` |
| `label` | `[1,1,0]` |

## Files

- `data/<split>-*.parquet` — one row per pair, `dry_audio` and `wet_audio` as playable `Audio`
  (the struct's `path` keeps the original WAV path); all other columns as in `metadata/<split>.csv`
- `data/<split>/dry/<source_sample_id>.wav` — dry clip (shared by its 7 wet versions)
- `data/<split>/wet/<sample_id>.wav` — processed audio
- `metadata/<split>.csv` — one row per pair: ids, paths (`dry_audio`, `wet_audio`), `effect_chain`,
  `label`, `eq_label` / `compressor_label` / `reverb_label`, exact parameters per effect
  (`eq_params`, `compressor_params`, `reverb_params`, JSON; empty when the effect is absent),
  normalised parameters, per-effect signal change and the per-sample seed
- `metadata/sources.csv` — how each dry clip was cut and normalised
- `metadata/dataset_summary.json`, `metadata/config_used.yaml` — full generation settings

## Usage

With 🤗 Datasets (`dry_audio` / `wet_audio` decode to arrays):

```python
import json
from datasets import load_dataset

ds = load_dataset("{config['huggingface']['repo_id']}")
row = ds["train"][0]
row["dry_audio"], row["wet_audio"]     # Audio
label = json.loads(row["label"])       # [EQ, Compressor, Reverb]
params = json.loads(row["parameters"])
```

Or directly from the WAV files and CSV metadata:

```python
import json, pandas as pd, soundfile as sf
from huggingface_hub import snapshot_download

root = snapshot_download("{config['huggingface']['repo_id']}", repo_type="dataset")
train = pd.read_csv(f"{{root}}/metadata/train.csv")
row = train.iloc[0]
dry, sr = sf.read(f"{{root}}/{{row.dry_audio}}")
wet, _ = sf.read(f"{{root}}/{{row.wet_audio}}")
label = json.loads(row.label)          # [EQ, Compressor, Reverb]
params = json.loads(row.parameters)    # {{"EQ": {{"frequency": ..., "gain": ..., "q": ...}}, ...}}
```

## License

Derived from MUSDB18 — **non-commercial research use only**, under the original dataset terms.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true", help="replace a previously generated dataset")
    parser.add_argument("--limit", type=int, help="only process the first N source samples (quick test)")
    args = parser.parse_args()

    config = load_config(args.config)
    s = generate(config, args.overwrite, args.limit)
    print("\n" + "=" * 64)
    print(f"Pairs generated : {s['num_pairs']} / {s['num_expected_pairs']} expected "
          f"({s['num_skipped_sources']} sources skipped, {s['num_failed_pairs']} failed)")
    for split, v in s["splits"].items():
        print(f"  {split:<10} songs={v['num_songs']:<3} ({v['song_fraction']:.0%})  "
              f"dry={v['num_sources']:<4} pairs={v['num_pairs']}")
    print(f"Labels          : {s['label_distribution']}")
    print(f"Low-change pairs: {s['num_low_effect_change_pairs']}")
    print(f"Output          : {config['paths']['output_dir']}")
    print("=" * 64)


if __name__ == "__main__":
    main()
