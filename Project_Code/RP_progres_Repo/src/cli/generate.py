"""CLI: generate the Phase 1 effect-type dataset.

Examples (from the repository root):

    # smoke test: 3 source files -> 21 pairs, then automatic validation
    python -m src.data.generate_dataset --input-dir /path/to/musdb/train --smoke-test

    # full dataset: 100 source files -> 700 pairs
    python -m src.data.generate_dataset --input-dir /path/to/musdb/train --num-source-files 100 --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from src.data.generator import DatasetGenerator, GenerationError
from src.data.metadata import LOG_FILE
from src.data.selection import SelectionError
from src.data.validation import validate_dataset
from src.effects import CANONICAL_ORDER
from src.effects.base import EffectError
from src.utils.config import (
    DEFAULT_CONFIG_PATH,
    ConfigError,
    load_config,
    resolve_config_path,
    with_overrides,
)
from src.utils.logging import setup_logging

logger = logging.getLogger("src.cli.generate")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.data.generate_dataset",
        description="Generate the Phase 1 dry/wet dataset for multi-label effect-type prediction "
                    "([EQ, Compressor, Reverb]). Settings come from the YAML config; flags override it.",
    )
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="YAML config (default: %(default)s)")
    p.add_argument("--input-dir", type=Path, help="Source dataset root, e.g. <musdb>/train (read only)")
    p.add_argument("--output-dir", type=Path, help="Where dry/wet audio is written")
    p.add_argument("--metadata-dir", type=Path, help="Where CSV/JSON metadata is written")
    p.add_argument("--num-source-files", type=int, help="Number of ORIGINAL dry source files to select")
    p.add_argument("--seed", type=int, help="Global random seed")
    p.add_argument("--sample-rate", type=int, help="Output sample rate in Hz")
    p.add_argument("--duration", type=float, help="Clip duration in seconds")
    p.add_argument("--channels", type=int, choices=(1, 2), help="1 = mono, 2 = stereo")
    p.add_argument("--parameter-mode", choices=("fixed", "random"), help="Effect-parameter generation mode")
    p.add_argument("--split-ratio", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                   help="Split ratios, e.g. 0.7 0.15 0.15")
    p.add_argument("--effects", nargs="+", choices=CANONICAL_ORDER,
                   help="Effects to enable (default: all three); all non-empty combinations are generated")
    p.add_argument("--backend", choices=("scipy", "pedalboard"), help="DSP backend for the effects")
    p.add_argument("--smoke-test", action="store_true",
                   help="Process a few source files (smoke_test.* in the config) into separate output folders, "
                        "then validate")
    p.add_argument("--overwrite", action="store_true", help="Replace files from a previous run in the output dirs")
    p.add_argument("--no-validate", action="store_true", help="Skip the validation step after generation")
    p.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), help="Logging verbosity")
    return p


def resolve_settings(args: argparse.Namespace) -> tuple[dict[str, Any], Path, Path, Path]:
    """Merge config + CLI flags and resolve the three directories."""
    config = load_config(args.config)
    overrides: dict[str, Any] = {
        "dataset.num_source_files": args.num_source_files,
        "generation.seed": args.seed,
        "audio.sample_rate": args.sample_rate,
        "audio.duration": args.duration,
        "audio.channels": args.channels,
        "generation.parameter_mode": args.parameter_mode,
        "effects.backend": args.backend,
        "logging.level": args.log_level,
    }
    if args.split_ratio:
        overrides.update({"split.train": args.split_ratio[0], "split.validation": args.split_ratio[1],
                          "split.test": args.split_ratio[2]})
    if args.effects:
        overrides["effects.enabled"] = {name: name in args.effects for name in CANONICAL_ORDER}

    output_cfg, metadata_cfg = config["paths"]["output_dir"], config["paths"]["metadata_dir"]
    if args.smoke_test:
        smoke = config.get("smoke_test", {})
        if args.num_source_files is None:
            overrides["dataset.num_source_files"] = int(smoke.get("num_source_files", 3))
        output_cfg = smoke.get("output_dir", output_cfg)
        metadata_cfg = smoke.get("metadata_dir", metadata_cfg)
    config = with_overrides(config, overrides)

    input_dir = args.input_dir.resolve() if args.input_dir else resolve_config_path(config, config["paths"]["input_dir"])
    if input_dir is None:
        raise ConfigError("No input directory given. Pass --input-dir or set paths.input_dir in the config.")
    output_dir = args.output_dir.resolve() if args.output_dir else resolve_config_path(config, output_cfg)
    metadata_dir = args.metadata_dir.resolve() if args.metadata_dir else resolve_config_path(config, metadata_cfg)

    # Record the directories actually used in the saved config.
    config = with_overrides(config, {"paths.input_dir": str(input_dir), "paths.output_dir": str(output_dir),
                                     "paths.metadata_dir": str(metadata_dir)})
    return config, input_dir, output_dir, metadata_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config, input_dir, output_dir, metadata_dir = resolve_settings(args)
    except ConfigError as exc:
        setup_logging("INFO")
        logger.error("%s", exc)
        return 2

    setup_logging(config.get("logging", {}).get("level", "INFO"), metadata_dir / LOG_FILE)
    mode = "SMOKE TEST" if args.smoke_test else "FULL RUN"
    logger.info("Phase 1 dataset generation (%s)", mode)
    logger.info("Config: %s", config["_config_path"])
    logger.info("Input: %s | Output: %s | Metadata: %s", input_dir, output_dir, metadata_dir)

    try:
        generator = DatasetGenerator(config)
        report = generator.run(input_dir, output_dir, metadata_dir, overwrite=args.overwrite)
    except (ConfigError, GenerationError, SelectionError, EffectError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1

    status = 0 if not report.failures else 1
    if not args.no_validate:
        logger.info("Validating generated dataset ...")
        validation = validate_dataset(output_dir, metadata_dir)
        for line in validation.format().splitlines():
            logger.info(line)
        if not validation.ok:
            status = 1
    if args.smoke_test and status == 0:
        logger.info("Smoke test passed. Run without --smoke-test to generate the full dataset.")
    return status


if __name__ == "__main__":
    sys.exit(main())
