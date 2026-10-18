"""Step 2 — generate the Phase 1 dry/wet effect dataset with Pedalboard.

For every source sample (e.g. ``1-Bass``) one dry clip is prepared and every
chain of ``generation.chain_mode`` is rendered:

* ``subset`` (config.yaml): the 7 combinations in the fixed order EQ -> Compressor -> Reverb

    1-Bass-EQ, 1-Bass-Compressor, 1-Bass-Reverb, 1-Bass-EQ-Compressor,
    1-Bass-EQ-Reverb, 1-Bass-Compressor-Reverb, 1-Bass-EQ-Compressor-Reverb

* ``ordered`` (config_order.yaml): all 15 orders, e.g. 1-Bass-Reverb-EQ;
  label = position of each effect in the chain. Built batch by batch with
  ``run_batches.py``.

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
from fx import (LABEL_COLUMNS, LABEL_NAMES, PARAM_COLUMNS, POSITION_COLUMNS, ParameterSampler, chain_name,
                effect_chains, label_vector, position_vector, render_chain, slot_vector)

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
ORDER_COLUMNS = (
    "presence_label", "eq_position", "compressor_position", "reverb_position", "slot_label",
    "chain_class", "num_effects",
)
SOURCE_COLUMNS = (
    "source_sample_id", "source_id", "source_category", "source_song", "split", "dry_audio", "source_file",
    "original_sample_rate", "original_channels", "original_duration_s", "clip_start_s", "clip_end_s",
    "active_fraction", "normalization", "input_loudness_lufs", "normalization_gain_db", "dry_peak_dbfs",
)
SKIPPED_COLUMNS = ("source_sample_id", "source_file", "split", "reason")
FAILURE_COLUMNS = ("sample_id", "source_sample_id", "effect_chain", "error")


LABEL_FORMATS = {
    "subset": "presence [EQ, Compressor, Reverb]: 1 = effect used",
    "ordered": "position [EQ, Compressor, Reverb]: 0 = absent, 1..3 = position in the chain",
}


def chain_mode(config: dict[str, Any]) -> str:
    return config["generation"].get("chain_mode", "subset")


def sample_columns(mode: str) -> tuple[str, ...]:
    """Subset mode keeps the original Phase 1 columns; ordered mode adds the order labels."""
    if mode == "subset":
        return SAMPLE_COLUMNS
    i = SAMPLE_COLUMNS.index("reverb_label") + 1
    return SAMPLE_COLUMNS[:i] + ORDER_COLUMNS + SAMPLE_COLUMNS[i:] + ("parameter_set_id",)


def dsp_backend() -> str:
    return f"pedalboard {pedalboard.__version__}"


def load_sources(config: dict[str, Any], source_dir: Path | None = None) -> list[dict[str, Any]]:
    """Source samples from the downloaded dataset, filtered and in a stable order."""
    source_dir = source_dir or config["paths"]["source_dir"]
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


def process_sources(config: dict[str, Any], sources: list[dict[str, Any]], source_dir: Path,
                    output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Dry clip + every chain for each source. Writes the WAVs and returns the metadata rows
    (``records``, ``source_rows``, ``skipped``, ``failures``). Works on any subset of sources, so
    a full run and a batch-by-batch run give identical files."""
    gen = config["generation"]
    seed = int(gen["seed"])
    order = list(config["effect_order"])
    if sorted(order) != sorted(LABEL_NAMES):
        sys.exit(f"effect_order must contain exactly {list(LABEL_NAMES)}")
    mode = chain_mode(config)
    chains = effect_chains(order, mode)
    # Ordered mode: every order of an effect set reuses the parameters of its canonical order.
    share = mode == "ordered" and bool(gen.get("share_parameters", False))
    sampler = ParameterSampler(config["parameters"], gen["parameter_mode"])
    backend = dsp_backend()
    subtype = config["audio"]["output_subtype"]

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
        write_audio(output_dir / dry_rel, clip.audio, clip.sample_rate, subtype)
        base = {"source_sample_id": src_sid, "source_id": int(src["source_id"]),
                "source_category": src["source_category"], "source_song": src["source_song"], "split": split}
        source_rows.append({**base, "dry_audio": dry_rel, "source_file": src["file_name"], **clip.info})

        set_params: dict[tuple[str, ...], dict] = {}  # canonical effect set -> its parameters
        for class_id, chain in enumerate(chains):
            canonical = tuple(e for e in order if e in chain)
            sample_id = f"{src_sid}-{'-'.join(chain)}"
            # Seeded by the canonical chain's id, so the canonical order renders exactly as in subset mode.
            param_set_id = f"{src_sid}-{'-'.join(canonical)}" if share else sample_id
            sample_seed = int(np.random.SeedSequence([seed, STREAM_PARAMS, stable_hash(param_set_id)]).generate_state(1)[0])
            fixed = set_params.get(canonical) if share and chain != canonical else None
            if share and chain != canonical and fixed is None:
                failures.append({"sample_id": sample_id, "source_sample_id": src_sid, "effect_chain": list(chain),
                                 "error": f"parameters of {chain_name(canonical, mode)} unavailable"})
                continue
            try:
                result = render_chain(clip.audio, clip.sample_rate, chain, sampler,
                                      np.random.default_rng(sample_seed),
                                      float(gen["min_effect_change_db"]), int(gen["max_retries"]), fixed)
            except Exception as exc:  # noqa: BLE001 — record and continue
                failures.append({"sample_id": sample_id, "source_sample_id": src_sid,
                                 "effect_chain": list(chain), "error": str(exc)})
                continue
            if share and chain == canonical:
                set_params[canonical] = result["parameters"]
            wet_rel = f"data/{split}/wet/{sample_id}.wav"
            write_audio(output_dir / wet_rel, result["wet"], clip.sample_rate, subtype)

            params = result["parameters"]
            presence = label_vector(chain)
            record = {
                "sample_id": sample_id, **base,
                "dry_audio": dry_rel, "wet_audio": wet_rel,
                "effect_chain": list(chain), "combination": chain_name(chain, mode),
                "label": presence if mode == "subset" else position_vector(chain),
                **{LABEL_COLUMNS[n]: v for n, v in zip(LABEL_NAMES, presence)},
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
            }
            if mode == "ordered":
                record.update({
                    "presence_label": presence,
                    **{POSITION_COLUMNS[n]: p for n, p in zip(LABEL_NAMES, position_vector(chain))},
                    "slot_label": slot_vector(chain), "chain_class": class_id, "num_effects": len(chain),
                    "parameter_set_id": param_set_id,
                })
            records.append(record)
        print(f"[{k}/{len(sources)}] {split:<10} {src_sid:<12} {src['source_song']}  ({time.time() - started:.0f}s)")
    return {"records": records, "source_rows": source_rows, "skipped": skipped, "failures": failures}


def generate(config: dict[str, Any], overwrite: bool, limit: int | None) -> dict[str, Any]:
    source_dir, output_dir = config["paths"]["source_dir"], config["paths"]["output_dir"]
    gen = config["generation"]
    sources = load_sources(config)
    if limit:
        sources = sources[:limit]
    prepare_output_dir(output_dir, source_dir, overwrite)
    n_chains = len(effect_chains(config["effect_order"], chain_mode(config)))
    print(f"{len(sources)} source samples x {n_chains} combinations = {len(sources) * n_chains} pairs "
          f"(seed {gen['seed']}, {gen['parameter_mode']} parameters, {dsp_backend()})")

    result = process_sources(config, sources, source_dir, output_dir)
    if not result["records"]:
        sys.exit("No samples were generated.")
    return write_metadata(config, **result)


def write_metadata(config, records, source_rows, skipped, failures, extra: dict | None = None) -> dict[str, Any]:
    """Metadata CSVs, summary and dataset card for all ``records`` (``extra`` is merged into the summary)."""
    output_dir = config["paths"]["output_dir"]
    meta = output_dir / "metadata"
    splits = config["splits"]
    mode = chain_mode(config)
    chains = effect_chains(config["effect_order"], mode)
    for split in splits:
        write_csv(meta / f"{split}.csv", [r for r in records if r["split"] == split], sample_columns(mode))
    write_csv(meta / "sources.csv", source_rows, SOURCE_COLUMNS)
    write_csv(meta / "skipped.csv", skipped, SKIPPED_COLUMNS)
    write_csv(meta / "failures.csv", failures, FAILURE_COLUMNS)
    public = {k: v for k, v in config.items() if k != "paths"}
    (meta / "config_used.yaml").write_text(yaml.safe_dump(public, sort_keys=False), encoding="utf-8")

    songs = {s: sorted({r["source_id"] for r in source_rows if r["split"] == s}) for s in splits}
    total_songs = sum(len(v) for v in songs.values())
    summary = {
        "phase": "phase1_effect_type" if mode == "subset" else "phase1_effect_order",
        "task": ("multi-label effect-type classification from paired dry/wet audio" if mode == "subset" else
                 "effect-chain prediction (which effects, in which order) from paired dry/wet audio"),
        "source_dataset": config["source"]["repo_id"],
        "label_names": list(LABEL_NAMES),
        "chain_mode": mode,
        "label_format": LABEL_FORMATS[mode],
        "processing_order": (" -> ".join(config["effect_order"]) if mode == "subset" else
                             "chain order (every order of every non-empty subset)"),
        "effect_combinations": [chain_name(c, mode) for c in chains],
        "sample_id_format": "<source_id>-<source_category>-<effect>[-<effect>...]",
        "num_sources": len(source_rows),
        "num_pairs": len(records),
        "num_expected_pairs": len(source_rows) * len(chains),
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
        "dsp_backend": dsp_backend(),
        "python": platform.python_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if mode == "ordered":
        summary["share_parameters"] = bool(config["generation"].get("share_parameters", False))
        summary["chain_classes"] = [
            {"chain_class": i, "chain": chain_name(c, mode), "label": position_vector(c),
             "presence_label": label_vector(c), "slot_label": slot_vector(c)}
            for i, c in enumerate(chains)
        ]
    summary.update(extra or {})
    (meta / SUMMARY_FILE).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(dataset_card(config, summary), encoding="utf-8")
    return summary


def dataset_card(config: dict[str, Any], s: dict[str, Any]) -> str:
    if s.get("chain_mode") == "ordered":
        return order_dataset_card(config, s)
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


def order_dataset_card(config: dict[str, Any], s: dict[str, Any]) -> str:
    """Dataset card for the 15-chain effect-order dataset (``chain_mode: ordered``)."""
    repo_id = config["huggingface"]["repo_id"]
    title = config["huggingface"].get("pretty_name", "Inverse FX Phase 1 — Effect-Order Dataset")
    split_rows = "\n".join(
        f"| {k} | {v['num_songs']} ({v['song_fraction']:.0%}) | {v['num_sources']} | {v['num_pairs']} |"
        for k, v in s["splits"].items()
    )
    chain_rows = "\n".join(
        f"| {c['chain_class']} | {c['chain'].replace('->', ' → ')} | `{c['label']}` | `{c['presence_label']}` | "
        f"`{c['slot_label']}` |"
        for c in s["chain_classes"]
    )
    a = s["audio"]
    # Only splits that already have shards, so the viewer never points at a missing file pattern.
    data_files = "\n".join(f"  - split: {k}\n    path: data/{k}-*.parquet"
                           for k, v in s["splits"].items() if v["num_pairs"])
    progress = s.get("batches")
    status = ""
    if progress and progress["completed"] < progress["total"]:
        status = (f"\n> **Build in progress:** batch {progress['completed']} of {progress['total']} uploaded "
                  f"({progress['songs_per_batch']} songs per batch). The counts below cover the uploaded batches.\n")
    shared = ("Every order of the same effect set uses the **same parameters** for a given dry clip, so e.g. "
              "`EQ → Reverb` and `Reverb → EQ` differ only in order (`parameter_set_id`)."
              if s.get("share_parameters") else "Every chain samples its own parameters.")
    return f"""---
pretty_name: {title}
license: other
license_name: musdb18
license_link: https://sigsep.github.io/datasets/musdb.html
task_categories:
- audio-classification
tags:
- audio
- music
- audio-effects
- effect-order
- pedalboard
configs:
- config_name: default
  data_files:
{data_files}
---

# {title}
{status}
Paired **dry / wet** audio for predicting **which effects were applied and in which order**
(Inverse FX Chain Predictor). Generated from
[{s['source_dataset']}](https://huggingface.co/datasets/{s['source_dataset']}) with
[Pedalboard](https://github.com/spotify/pedalboard) ({s['dsp_backend']}).

- Effects: **EQ**, **Compressor**, **Reverb** — every order of every non-empty subset,
  **{len(s['chain_classes'])} chains** per dry clip, applied in the chain's own order
- Label: position vector `[EQ, Compressor, Reverb]` — `0` = absent, `1`/`2`/`3` = position in the chain
- {shared}
- Sources: {s['num_sources']} dry clips ({', '.join(f'{k}: {v}' for k, v in s['category_distribution'].items())})
- Pairs: {s['num_pairs']}
- Audio: {a['sample_rate']} Hz, {a['channels']} channel(s), {a['duration']} s clips, 32-bit float WAV
- Effect parameters: {s['parameter_mode']}, reproducible with seed {s['random_seed']}

## Chains and labels

| chain_class | chain | `label` (position) | `presence_label` | `slot_label` |
|---|---|---|---|---|
{chain_rows}

| column | meaning |
|---|---|
| `label`, `eq_position` / `compressor_position` / `reverb_position` | position of each effect, 0 = absent |
| `presence_label`, `eq_label` / `compressor_label` / `reverb_label` | 1 if the effect is used |
| `slot_label` | effect at each slot: 1 = EQ, 2 = Compressor, 3 = Reverb, 0 = empty |
| `chain_class` | 0–{len(s['chain_classes']) - 1}, row of the table above (single-label classification) |
| `num_effects` | chain length (1–3) |
| `parameter_set_id` | rows with the same id use identical effect parameters |

## Splits (song level, inherited from the source dataset)

| split | songs | dry clips | dry/wet pairs |
|---|---|---|---|
{split_rows}

## Naming

| field | example |
|---|---|
| `source_sample_id` | `1-Bass` |
| `sample_id` | `1-Bass-Reverb-EQ` (effects in processing order) |
| `effect_chain` | `["Reverb","EQ"]` |
| `combination` | `Reverb->EQ` |
| `label` | `[2,0,1]` |

## Files

- `data/<split>-*.parquet` — one row per pair, `dry_audio` and `wet_audio` as playable `Audio`;
  all other columns as in `metadata/<split>.csv`
- `data/<split>/dry/<source_sample_id>.wav` — dry clip (shared by its {len(s['chain_classes'])} wet versions)
- `data/<split>/wet/<sample_id>.wav` — processed audio
- `metadata/<split>.csv` — one row per pair (lists/dicts are JSON strings), exact parameters per effect
- `metadata/sources.csv`, `skipped.csv`, `failures.csv`, `dataset_summary.json`, `config_used.yaml`

## Usage

```python
import json
from datasets import load_dataset

ds = load_dataset("{repo_id}")
row = ds["train"][0]
row["dry_audio"], row["wet_audio"]          # Audio
position = json.loads(row["label"])         # e.g. [2, 0, 1] -> Reverb, then EQ
chain = json.loads(row["effect_chain"])     # ["Reverb", "EQ"]
row["chain_class"]                          # 0..{len(s['chain_classes']) - 1}
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
