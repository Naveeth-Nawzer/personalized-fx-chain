"""Step 1 — download the clean source dataset from the Hugging Face Hub.

The files go to ``paths.source_dir`` (default ``data/source``). Nothing is
ever written back to the source repository. Re-running only fetches files
that changed.

Usage::

    .venv\\Scripts\\python download_source.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from common import DEFAULT_CONFIG, load_config, read_csv, read_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    repo_id = config["source"]["repo_id"]
    target = config["paths"]["source_dir"]
    print(f"Downloading {repo_id} -> {target}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
        token=read_token(config["paths"]["env_file"]),
        allow_patterns=["data/**", "metadata/**", "README.md"],
    )

    samples = read_csv(target / "metadata" / "samples.csv")
    missing = [s["file_name"] for s in samples if not (target / s["file_name"]).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} audio file(s) listed in samples.csv are missing, e.g. {missing[:3]}")
    print(f"OK: {len(samples)} source samples available")


if __name__ == "__main__":
    main()
