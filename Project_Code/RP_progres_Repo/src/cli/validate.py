"""CLI: validate a generated Phase 1 dataset.

    python -m src.data.validate_dataset                      # default (full) dataset
    python -m src.data.validate_dataset --smoke-test         # smoke-test dataset
    python -m src.data.validate_dataset --output-dir D --metadata-dir M
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from src.data.validation import validate_dataset
from src.utils.config import DEFAULT_CONFIG_PATH, ConfigError, load_config, resolve_config_path
from src.utils.logging import setup_logging

logger = logging.getLogger("src.cli.validate")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.data.validate_dataset",
                                description="Validate generated Phase 1 dry/wet audio and metadata.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config (default: %(default)s)")
    p.add_argument("--output-dir", type=Path, help="Dataset audio root (default: from config)")
    p.add_argument("--metadata-dir", type=Path, help="Metadata directory (default: from config)")
    p.add_argument("--smoke-test", action="store_true", help="Validate the smoke-test dataset locations")
    p.add_argument("--skip-audio", action="store_true", help="Only check metadata and file existence (fast)")
    p.add_argument("--max-issues", type=int, default=20, help="Issues listed per check (default: %(default)s)")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging("INFO")
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        logger.error("%s", exc)
        return 2
    section = config.get("smoke_test", {}) if args.smoke_test else config["paths"]
    output_dir = args.output_dir.resolve() if args.output_dir else resolve_config_path(config, section["output_dir"])
    metadata_dir = (args.metadata_dir.resolve() if args.metadata_dir
                    else resolve_config_path(config, section["metadata_dir"]))
    logger.info("Validating dataset: audio=%s metadata=%s", output_dir, metadata_dir)
    report = validate_dataset(output_dir, metadata_dir, check_audio=not args.skip_audio)
    print(report.format(max_per_check=args.max_issues))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
