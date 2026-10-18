"""Step 3 — validate the generated Phase 1 dataset before pushing.

Checks:
  * every source has all chains (7 in subset mode, 15 in ordered mode) and every audio file exists
  * sample_id = <source_sample_id>-<effects joined by '-'>
  * label / eq_label / compressor_label / reverb_label agree with effect_chain
    (ordered mode: also positions, presence_label, slot_label, chain_class, num_effects)
  * parameters are stored exactly for the effects in the chain (and only those)
  * ordered mode with shared parameters: every order of an effect set has identical parameters
  * no song (source_id) appears in more than one split
  * dry and wet audio have the same sample rate and length, finite samples, wet != dry
  * re-rendering a random subset from the stored parameters reproduces the wet audio

Usage::

    .venv\\Scripts\\python validate_phase1.py
    .venv\\Scripts\\python validate_phase1.py --rerender 50
    .venv\\Scripts\\python validate_phase1.py --metadata-only   # audio not present (batch build)
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from common import DEFAULT_CONFIG, load_config, read_csv
from fx import (LABEL_COLUMNS, LABEL_NAMES, PARAM_COLUMNS, POSITION_COLUMNS, apply_effect, chain_name,
                effect_chains, label_vector, position_vector, slot_vector)


def load_rows(root: Path, splits: list[str]) -> list[dict[str, str]]:
    rows = []
    for split in splits:
        path = root / "metadata" / f"{split}.csv"
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run generate_phase1.py first.")
        rows += read_csv(path)
    return rows


def check_metadata(rows: list[dict[str, str]], config: dict[str, Any]) -> list[str]:
    """Ids, labels, parameters, chain coverage and split leakage (no audio needed)."""
    order = list(config["effect_order"])
    mode = config["generation"].get("chain_mode", "subset")
    chains = effect_chains(order, mode)
    names = {chain_name(c, mode) for c in chains}
    share = mode == "ordered" and bool(config["generation"].get("share_parameters", False))
    errors: list[str] = []

    by_source: dict[str, set[str]] = defaultdict(set)
    splits_of_song: dict[str, set[str]] = defaultdict(set)
    param_sets: dict[str, dict] = {}
    for r in rows:
        sid, chain = r["sample_id"], json.loads(r["effect_chain"])
        by_source[r["source_sample_id"]].add(r["combination"])
        splits_of_song[r["source_id"]].add(r["split"])
        if r["source_sample_id"] != f"{r['source_id']}-{r['source_category']}":
            errors.append(f"{sid}: source_sample_id does not match <source_id>-<source_category>")
        if sid != f"{r['source_sample_id']}-{'-'.join(chain)}":
            errors.append(f"{sid}: sample_id does not match its effect_chain {chain}")
        if tuple(chain) not in chains:
            errors.append(f"{sid}: effect_chain {chain} is not a valid {mode} chain")
            continue
        if r["combination"] != chain_name(chain, mode):
            errors.append(f"{sid}: combination {r['combination']} != {chain_name(chain, mode)}")
        presence = label_vector(chain)
        expected = presence if mode == "subset" else position_vector(chain)
        if json.loads(r["label"]) != expected:
            errors.append(f"{sid}: label {r['label']} != {expected}")
        if mode == "ordered":
            checks = {"presence_label": (json.loads(r["presence_label"]), presence),
                      "slot_label": (json.loads(r["slot_label"]), slot_vector(chain)),
                      "chain_class": (int(r["chain_class"]), chains.index(tuple(chain))),
                      "num_effects": (int(r["num_effects"]), len(chain))}
            checks.update({POSITION_COLUMNS[n]: (int(r[POSITION_COLUMNS[n]]), p)
                           for n, p in zip(LABEL_NAMES, position_vector(chain))})
            errors += [f"{sid}: {col} {got} != {want}" for col, (got, want) in checks.items() if got != want]
        for name, value in zip(LABEL_NAMES, presence):
            if int(r[LABEL_COLUMNS[name]]) != value:
                errors.append(f"{sid}: {LABEL_COLUMNS[name]} != {value}")
            stored = json.loads(r[PARAM_COLUMNS[name]]) if r[PARAM_COLUMNS[name]] else None
            if value and (stored is None or set(stored) != set(config["parameters"][name])):
                errors.append(f"{sid}: {PARAM_COLUMNS[name]} missing or incomplete")
            if not value and stored is not None:
                errors.append(f"{sid}: {PARAM_COLUMNS[name]} set for an absent effect")
        if share:
            canonical = [e for e in order if e in chain]
            set_id = f"{r['source_sample_id']}-{'-'.join(canonical)}"
            if r["parameter_set_id"] != set_id:
                errors.append(f"{sid}: parameter_set_id {r['parameter_set_id']} != {set_id}")
            params = json.loads(r["parameters"])
            if param_sets.setdefault(set_id, params) != params:
                errors.append(f"{sid}: parameters differ from other orders of {set_id}")

    for src, found in by_source.items():
        if found != names:
            errors.append(f"{src}: has {len(found)} chains, missing {sorted(names - found)}")
    for song, splits in splits_of_song.items():
        if len(splits) > 1:
            errors.append(f"source_id {song} leaks across splits {sorted(splits)}")
    return errors


def check_audio(rows: list[dict[str, str]], root: Path, rerender: int) -> tuple[list[str], int, float]:
    """Audio checks for every row and re-rendering of a random subset.
    Returns ``(errors, number re-rendered, max abs difference)``."""
    errors: list[str] = []
    dry_cache: dict[str, tuple[np.ndarray, int]] = {}
    pick = set(random.Random(0).sample(range(len(rows)), min(rerender, len(rows))))
    max_diff = 0.0
    for i, r in enumerate(rows):
        missing = [col for col in ("dry_audio", "wet_audio") if not (root / r[col]).is_file()]
        if missing:
            errors += [f"{r['sample_id']}: missing file {r[col]}" for col in missing]
            continue
        if r["dry_audio"] not in dry_cache:
            dry_cache[r["dry_audio"]] = sf.read(root / r["dry_audio"], dtype="float32", always_2d=True)
        dry, sr = dry_cache[r["dry_audio"]]
        wet, wsr = sf.read(root / r["wet_audio"], dtype="float32", always_2d=True)
        if sr != wsr or dry.shape != wet.shape:
            errors.append(f"{r['sample_id']}: dry/wet mismatch (sr {sr}/{wsr}, shape {dry.shape}/{wet.shape})")
            continue
        if not np.all(np.isfinite(wet)) or np.array_equal(dry, wet):
            errors.append(f"{r['sample_id']}: wet audio is non-finite or identical to dry")
        if i in pick:
            x, params = dry.T, json.loads(r["parameters"])
            for effect in json.loads(r["effect_chain"]):
                x = apply_effect(effect, x, sr, params[effect])
            diff = float(np.max(np.abs(x.T - wet)))
            max_diff = max(max_diff, diff)
            if diff > 1e-4:
                errors.append(f"{r['sample_id']}: re-render differs from saved wet audio (max {diff:.2e})")
    return errors, len(pick), max_diff


def report(errors: list[str]) -> None:
    if errors:
        print(f"FAILED with {len(errors)} problem(s):")
        for e in errors[:50]:
            print("  -", e)
        raise SystemExit(1)
    print("All checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rerender", type=int, default=20, help="pairs to re-render and compare (0 = skip)")
    parser.add_argument("--metadata-only", action="store_true", help="skip audio checks (audio not on disk)")
    args = parser.parse_args()

    config = load_config(args.config)
    root = config["paths"]["output_dir"]
    rows = load_rows(root, config["splits"])
    errors = check_metadata(rows, config)
    sources = len({r["source_sample_id"] for r in rows})
    if args.metadata_only:
        print(f"Checked metadata of {len(rows)} pairs from {sources} sources")
    else:
        audio_errors, n, max_diff = check_audio(rows, root, args.rerender)
        errors += audio_errors
        print(f"Checked {len(rows)} pairs from {sources} sources; re-rendered {n} (max abs diff {max_diff:.2e})")
    report(errors)


if __name__ == "__main__":
    main()
