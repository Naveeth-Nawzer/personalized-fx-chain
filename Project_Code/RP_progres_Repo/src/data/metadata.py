"""Metadata records, CSV I/O and the dataset summary.

One CSV per split (``train.csv``, ``validation.csv``, ``test.csv``), one row per
dry/wet pair. Nested fields (lists/dicts) are stored as JSON strings. Paths
are relative to the dataset output directory and always use ``/``.
"""

from __future__ import annotations

import csv
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from src.data.labels import LABEL_NAMES
from src.utils.config import SPLITS

CSV_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "source_id",
    "source_file",
    "source_song",
    "source_category",
    "split",
    "dry_path",
    "wet_path",
    "effect_chain",
    "processing_order",
    "combination",
    "label",
    "eq_label",
    "compressor_label",
    "reverb_label",
    "parameters",
    "parameters_normalized",
    "effect_change_db",
    "low_effect_change",
    "sample_rate",
    "num_channels",
    "duration_s",
    "wet_peak_dbfs",
    "wet_clipped",
    "dsp_backend",
    "parameter_mode",
    "random_seed",
    "sample_seed",
    "preprocessing",
)
JSON_COLUMNS = frozenset(
    {"effect_chain", "processing_order", "label", "parameters", "parameters_normalized",
     "effect_change_db", "low_effect_change", "preprocessing"}
)
INT_COLUMNS = frozenset(
    {"eq_label", "compressor_label", "reverb_label", "sample_rate", "num_channels",
     "wet_clipped", "random_seed", "sample_seed"}
)
FLOAT_COLUMNS = frozenset({"duration_s", "wet_peak_dbfs"})

SELECTED_SOURCES_COLUMNS = (
    "source_id", "split", "source_file", "source_song", "source_category", "preprocessing",
)
SKIPPED_COLUMNS = ("source_file", "source_song", "source_category", "reason")
FAILURE_COLUMNS = ("stage", "source_id", "source_file", "sample_id", "effect_chain", "error")

SUMMARY_FILE = "dataset_summary.json"
SELECTED_SOURCES_FILE = "selected_sources.csv"
SKIPPED_FILE = "skipped_files.csv"
FAILURES_FILE = "failures.csv"
CONFIG_USED_FILE = "config_used.yaml"
LOG_FILE = "generation.log"
#: Every file the generator writes into the metadata directory.
OWNED_METADATA_FILES = tuple(f"{s}.csv" for s in SPLITS) + (
    SUMMARY_FILE, SELECTED_SOURCES_FILE, SKIPPED_FILE, FAILURES_FILE, CONFIG_USED_FILE, LOG_FILE,
)


def _dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=False, separators=(",", ":"))


def record_to_row(record: Mapping[str, Any], columns: Sequence[str] = CSV_COLUMNS) -> dict[str, str]:
    """Serialise a record for CSV (nested values as JSON)."""
    row: dict[str, str] = {}
    for col in columns:
        value = record.get(col)
        if isinstance(value, (list, dict, tuple)) or (col in JSON_COLUMNS and value is not None):
            row[col] = _dumps(value)
        elif value is None:
            row[col] = ""
        elif isinstance(value, bool):
            row[col] = str(int(value))
        else:
            row[col] = str(value)
    return row


def row_to_record(row: Mapping[str, str]) -> dict[str, Any]:
    """Parse a CSV row back into typed values (inverse of :func:`record_to_row`)."""
    record: dict[str, Any] = {}
    for col, raw in row.items():
        if col in JSON_COLUMNS:
            record[col] = json.loads(raw) if raw else None
        elif col in INT_COLUMNS:
            record[col] = int(raw) if raw != "" else None
        elif col in FLOAT_COLUMNS:
            record[col] = float(raw) if raw != "" else None
        else:
            record[col] = raw
    return record


def write_table(path: Path, records: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> Path:
    """Write records to a CSV file with a fixed column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        for record in records:
            writer.writerow(record_to_row(record, columns))
    return path


def read_table(path: Path) -> list[dict[str, Any]]:
    """Read a CSV written by :func:`write_table`."""
    with path.open("r", newline="", encoding="utf-8") as fh:
        return [row_to_record(row) for row in csv.DictReader(fh)]


def write_split_csvs(records: Sequence[Mapping[str, Any]], metadata_dir: Path) -> dict[str, Path]:
    """Write ``train.csv`` / ``validation.csv`` / ``test.csv`` (all three, even if empty)."""
    return {
        split: write_table(metadata_dir / f"{split}.csv", [r for r in records if r["split"] == split], CSV_COLUMNS)
        for split in SPLITS
    }


def read_split_csvs(metadata_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the per-split CSVs. Missing files raise ``FileNotFoundError``."""
    result = {}
    for split in SPLITS:
        path = metadata_dir / f"{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Metadata file not found: {path}")
        result[split] = read_table(path)
    return result


def write_json(path: Path, data: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def write_yaml(path: Path, data: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(data), sort_keys=False), encoding="utf-8")
    return path


def library_versions() -> dict[str, str]:
    """Versions of Python and the libraries that influence the generated data."""
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "soundfile", "pyloudnorm", "PyYAML", "pedalboard"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def build_summary(
    records: Sequence[Mapping[str, Any]],
    selected_sources: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    combinations: Sequence[Sequence[str]],
    *,
    input_dir_name: str,
    num_discovered: int,
    num_skipped: int,
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble ``dataset_summary.json``."""
    per_split: dict[str, Any] = {}
    for split in SPLITS:
        split_records = [r for r in records if r["split"] == split]
        split_sources = [s for s in selected_sources if s["split"] == split]
        per_split[split] = {
            "num_source_files": len(split_sources),
            "num_pairs": len(split_records),
            "num_songs": len({s["source_song"] for s in split_sources}),
            "source_category_distribution": dict(sorted(Counter(s["source_category"] for s in split_sources).items())),
        }

    return {
        "phase": "phase1_effect_type",
        "task": "multi-label effect-type classification from paired dry/wet audio",
        "label_names": list(LABEL_NAMES),
        "num_source_files": len(selected_sources),
        "num_generated_pairs": len(records),
        "num_expected_pairs": len(selected_sources) * len(combinations),
        "num_failed_generations": len(failures),
        "num_train_pairs": per_split["train"]["num_pairs"],
        "num_validation_pairs": per_split["validation"]["num_pairs"],
        "num_test_pairs": per_split["test"]["num_pairs"],
        "splits": per_split,
        "split_ratios": {s: config["split"][s] for s in SPLITS},
        "split_unit": "song (all stems of a song stay in one split)",
        "effect_combinations": ["+".join(c) for c in combinations],
        "effect_distribution": {
            name: sum(1 for r in records if name in r["effect_chain"]) for name in LABEL_NAMES
        },
        "combination_distribution": dict(Counter(r["combination"] for r in records)),
        "source_category_distribution": dict(sorted(Counter(s["source_category"] for s in selected_sources).items())),
        "audio": {
            "sample_rate": config["audio"]["sample_rate"],
            "channels": config["audio"]["channels"],
            "duration_s": config["audio"]["duration"],
            "format": config["audio"]["output_format"],
            "subtype": config["audio"]["output_subtype"],
        },
        "preprocessing": dict(config["preprocessing"]),
        "processing_order": "canonical: EQ -> Compressor -> Reverb (not a Phase 1 label)",
        "dsp_backend": config["effects"]["backend"],
        "dsp_backend_options": dict(config["effects"].get("scipy_reverb", {}))
        if config["effects"]["backend"] == "scipy" else {},
        "parameter_mode": config["generation"]["parameter_mode"],
        "min_effect_change_db": config["generation"]["min_effect_change_db"],
        "num_low_effect_change_samples": sum(1 for r in records if r.get("low_effect_change")),
        "random_seed": config["generation"]["seed"],
        "input_dir_name": input_dir_name,
        "num_discovered_files": num_discovered,
        "num_skipped_files": num_skipped,
        "generation_timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "software_versions": library_versions(),
    }
