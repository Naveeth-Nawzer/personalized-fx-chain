"""Shared helpers: config loading, HF token, CSV I/O, stable seeds."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
SUMMARY_FILE = "dataset_summary.json"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Read config.yaml and resolve every entry of ``paths`` against the config's folder."""
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["paths"] = {k: (path.parent / v).resolve() for k, v in config["paths"].items()}
    return config


def read_token(env_file: Path) -> str:
    """HF_TOKEN from the environment, else from a KEY=VALUE .env file (spaces/quotes tolerated)."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and Path(env_file).is_file():
        for line in Path(env_file).read_text(encoding="utf-8-sig").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().removeprefix("export ").strip() == "HF_TOKEN":
                token = value.strip().strip("'\"")
    if not token:
        sys.exit(f"HF_TOKEN not found in the environment or in {env_file}")
    return token


def stable_hash(text: str) -> int:
    """32-bit hash of a string that is identical on every machine and Python run."""
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, bool):
        return str(int(value))
    return str(value)


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Write rows with a fixed column order; lists/dicts are stored as JSON strings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_cell(row.get(c)) for c in columns])


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
