"""Step 3 — validate the generated Phase 1 dataset before pushing.

Checks:
  * every source has all 7 combinations and every audio file exists
  * sample_id = <source_sample_id>-<effects joined by '-'>
  * label / eq_label / compressor_label / reverb_label agree with effect_chain
  * parameters are stored exactly for the effects in the chain (and only those)
  * no song (source_id) appears in more than one split
  * dry and wet audio have the same sample rate and length, finite samples, wet != dry
  * re-rendering a random subset from the stored parameters reproduces the wet audio

Usage::

    .venv\\Scripts\\python validate_phase1.py
    .venv\\Scripts\\python validate_phase1.py --rerender 50
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from common import DEFAULT_CONFIG, load_config, read_csv
from fx import LABEL_COLUMNS, LABEL_NAMES, PARAM_COLUMNS, apply_effect, effect_combinations, label_vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--rerender", type=int, default=20, help="pairs to re-render and compare (0 = skip)")
    args = parser.parse_args()

    config = load_config(args.config)
    root = config["paths"]["output_dir"]
    order = list(config["effect_order"])
    combos = {"+".join(c) for c in effect_combinations(order)}
    errors: list[str] = []

    rows = []
    for split in config["splits"]:
        path = root / "metadata" / f"{split}.csv"
        if not path.is_file():
            raise SystemExit(f"Missing {path}. Run generate_phase1.py first.")
        rows += read_csv(path)

    by_source: dict[str, set[str]] = defaultdict(set)
    splits_of_song: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        sid, chain = r["sample_id"], json.loads(r["effect_chain"])
        by_source[r["source_sample_id"]].add(r["combination"])
        splits_of_song[r["source_id"]].add(r["split"])
        if r["source_sample_id"] != f"{r['source_id']}-{r['source_category']}":
            errors.append(f"{sid}: source_sample_id does not match <source_id>-<source_category>")
        if sid != f"{r['source_sample_id']}-{'-'.join(chain)}":
            errors.append(f"{sid}: sample_id does not match its effect_chain {chain}")
        if chain != [e for e in order if e in chain]:
            errors.append(f"{sid}: effect_chain {chain} is not in processing order {order}")
        expected = label_vector(chain)
        if json.loads(r["label"]) != expected:
            errors.append(f"{sid}: label {r['label']} != {expected}")
        for name, value in zip(LABEL_NAMES, expected):
            if int(r[LABEL_COLUMNS[name]]) != value:
                errors.append(f"{sid}: {LABEL_COLUMNS[name]} != {value}")
            stored = json.loads(r[PARAM_COLUMNS[name]]) if r[PARAM_COLUMNS[name]] else None
            if value and (stored is None or set(stored) != set(config["parameters"][name])):
                errors.append(f"{sid}: {PARAM_COLUMNS[name]} missing or incomplete")
            if not value and stored is not None:
                errors.append(f"{sid}: {PARAM_COLUMNS[name]} set for an absent effect")
        for col in ("dry_audio", "wet_audio"):
            if not (root / r[col]).is_file():
                errors.append(f"{sid}: missing file {r[col]}")

    for src, found in by_source.items():
        if found != combos:
            errors.append(f"{src}: has {len(found)} combinations, missing {sorted(combos - found)}")
    for song, splits in splits_of_song.items():
        if len(splits) > 1:
            errors.append(f"source_id {song} leaks across splits {sorted(splits)}")

    # Audio checks (every pair) and re-rendering (random subset).
    dry_cache: dict[str, tuple[np.ndarray, int]] = {}
    pick = set(random.Random(0).sample(range(len(rows)), min(args.rerender, len(rows))))
    max_diff = 0.0
    for i, r in enumerate(rows):
        if not ((root / r["dry_audio"]).is_file() and (root / r["wet_audio"]).is_file()):
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

    print(f"Checked {len(rows)} pairs from {len(by_source)} sources; "
          f"re-rendered {len(pick)} (max abs diff {max_diff:.2e})")
    if errors:
        print(f"FAILED with {len(errors)} problem(s):")
        for e in errors[:50]:
            print("  -", e)
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
