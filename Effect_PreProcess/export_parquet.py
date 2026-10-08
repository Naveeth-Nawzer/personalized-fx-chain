"""Step 3b — export Parquet shards so the Hub viewer shows playable audio.

Reads ``metadata/<split>.csv`` and the WAV files of a generated dataset and
writes ``data/<split>-NNNNN-of-NNNNN.parquet`` next to them, where
``dry_audio`` and ``wet_audio`` are Hugging Face ``Audio`` columns
(``{bytes, path}``: the original WAV bytes plus the original relative path).
Every other column keeps its values and the types the viewer inferred from
the CSVs. Nothing else is changed: WAV files and CSVs stay where they are.

The README dataset card is rewritten so its ``configs`` point at the shards.
After writing, every shard is read back and compared with the CSV + WAV files.

Usage::

    .venv\\Scripts\\python export_parquet.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common import DEFAULT_CONFIG, SUMMARY_FILE, load_config, read_csv
from generate_phase1 import SAMPLE_COLUMNS, dataset_card

AUDIO_COLUMNS = {"dry_audio", "wet_audio"}
INT_COLUMNS = {"source_id", "eq_label", "compressor_label", "reverb_label", "sample_rate", "num_channels",
               "random_seed", "sample_seed"}
FLOAT_COLUMNS = {"duration_s", "dry_peak_dbfs", "wet_peak_dbfs"}
AUDIO_TYPE = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
MAX_SHARD_BYTES = 300 * 1024**2   # keeps each file comfortably below Hub/viewer limits
ROW_GROUP_SIZE = 10               # small row groups so the viewer can page through audio


def column_spec(name: str) -> tuple[pa.DataType, dict]:
    """Arrow type + Hugging Face feature for one column."""
    if name in AUDIO_COLUMNS:
        return AUDIO_TYPE, {"_type": "Audio"}
    if name in INT_COLUMNS:
        return pa.int64(), {"dtype": "int64", "_type": "Value"}
    if name in FLOAT_COLUMNS:
        return pa.float64(), {"dtype": "float64", "_type": "Value"}
    return pa.string(), {"dtype": "string", "_type": "Value"}


def build_schema() -> pa.Schema:
    fields, features = [], {}
    for name in SAMPLE_COLUMNS:
        arrow_type, feature = column_spec(name)
        fields.append(pa.field(name, arrow_type))
        features[name] = feature
    # Same metadata key the `datasets` library writes; the Hub reads features from it.
    meta = {b"huggingface": json.dumps({"info": {"features": features}}).encode("utf-8")}
    return pa.schema(fields, metadata=meta)


def convert_row(row: dict[str, str], root: Path) -> dict:
    out = {}
    for name in SAMPLE_COLUMNS:
        raw = row[name]
        if name in AUDIO_COLUMNS:
            out[name] = {"bytes": (root / raw).read_bytes(), "path": raw}
        elif name in INT_COLUMNS:
            out[name] = int(raw)
        elif name in FLOAT_COLUMNS:
            out[name] = float(raw)
        else:
            out[name] = raw if raw != "" else None  # empty CSV cell = null (as the viewer showed it)
    return out


def export_split(split: str, rows: list[dict[str, str]], root: Path, schema: pa.Schema) -> list[Path]:
    sizes = [(root / r["dry_audio"]).stat().st_size + (root / r["wet_audio"]).stat().st_size for r in rows]
    num_shards = max(1, math.ceil(sum(sizes) / MAX_SHARD_BYTES))
    per_shard = math.ceil(len(rows) / num_shards)
    paths = []
    for i in range(num_shards):
        chunk = rows[i * per_shard : (i + 1) * per_shard]
        path = root / "data" / f"{split}-{i:05d}-of-{num_shards:05d}.parquet"
        table = pa.Table.from_pylist([convert_row(r, root) for r in chunk], schema=schema)
        pq.write_table(table, path, row_group_size=ROW_GROUP_SIZE)
        paths.append(path)
    return paths


def verify_split(paths: list[Path], rows: list[dict[str, str]], root: Path) -> None:
    """Read the shards back and compare every value with the CSV and every audio blob with its WAV file."""
    restored = [r for p in paths for r in pq.read_table(p).to_pylist()]
    if len(restored) != len(rows):
        raise SystemExit(f"Row count mismatch: {len(restored)} in Parquet vs {len(rows)} in CSV")
    for got, want in zip(restored, rows):
        for name in SAMPLE_COLUMNS:
            if name in AUDIO_COLUMNS:
                ok = got[name]["path"] == want[name] and got[name]["bytes"] == (root / want[name]).read_bytes()
            elif name in FLOAT_COLUMNS:
                ok = got[name] == float(want[name])
            else:
                ok = ("" if got[name] is None else str(got[name])) == want[name]
            if not ok:
                raise SystemExit(f"Parquet value differs from source for {want['sample_id']}.{name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    root = config["paths"]["output_dir"]
    summary_path = root / "metadata" / SUMMARY_FILE
    if not summary_path.is_file():
        raise SystemExit(f"No generated dataset at {root}. Run generate_phase1.py first.")

    for old in (root / "data").glob("*.parquet"):
        old.unlink()
    schema = build_schema()
    for split in config["splits"]:
        rows = read_csv(root / "metadata" / f"{split}.csv")
        paths = export_split(split, rows, root, schema)
        verify_split(paths, rows, root)
        size_mb = sum(p.stat().st_size for p in paths) / 1024**2
        print(f"{split:<10} {len(rows):>4} rows -> {len(paths)} shard(s), {size_mb:.0f} MB, verified")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    (root / "README.md").write_text(dataset_card(config, summary), encoding="utf-8")
    print("README.md configs now point at data/<split>-*.parquet")


if __name__ == "__main__":
    main()
