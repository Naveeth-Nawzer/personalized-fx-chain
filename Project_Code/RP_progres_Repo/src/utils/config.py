"""Configuration loading, merging and validation.

The YAML file in ``configs/phase1.yaml`` is the single source of truth for
pipeline settings. CLI arguments override individual keys on top of it.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "configs" / "phase1.yaml"

SPLITS: tuple[str, ...] = ("train", "validation", "test")
VALID_PARAMETER_MODES = ("fixed", "random")
VALID_NORMALIZATION = ("none", "peak", "lufs")
VALID_SUBTYPES = ("FLOAT", "PCM_24", "PCM_16")
VALID_SCALES = ("linear", "log")


class ConfigError(ValueError):
    """Raised when the configuration is missing a key or holds an invalid value."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Config file path. Defaults to ``configs/phase1.yaml``.

    Returns:
        The parsed configuration, with ``_config_path`` and ``_project_root``
        bookkeeping keys added so relative paths can be resolved later.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"Config file {config_path} must contain a mapping at the top level")
    config_path = config_path.resolve()
    config["_config_path"] = str(config_path)
    # The project root is the parent of the `configs/` directory.
    config["_project_root"] = str(config_path.parent.parent)
    return config


def get(config: Mapping[str, Any], dotted_key: str) -> Any:
    """Return ``config['a']['b']`` for ``dotted_key='a.b'``, raising ConfigError if missing."""
    node: Any = config
    for part in dotted_key.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"Missing config key: '{dotted_key}'")
        node = node[part]
    return node


def set_value(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``config['a']['b'] = value`` for ``dotted_key='a.b'``, creating sections as needed."""
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def with_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``config`` with dotted-key overrides applied (``None`` values are skipped)."""
    merged = copy.deepcopy(dict(config))
    for key, value in overrides.items():
        if value is not None:
            set_value(merged, key, value)
    return merged


def resolve_config_path(config: Mapping[str, Any], value: str | Path | None) -> Path | None:
    """Resolve a path from the config file relative to the project root."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        root = Path(config.get("_project_root", PROJECT_ROOT))
        path = root / path
    return path.resolve()


def enabled_effects(config: Mapping[str, Any]) -> list[str]:
    """Names of enabled effects, as written in the config."""
    enabled = get(config, "effects.enabled")
    return [name for name, flag in enabled.items() if flag]


def public_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Copy of the config without internal bookkeeping keys (safe to write to disk)."""
    return {k: copy.deepcopy(v) for k, v in config.items() if not k.startswith("_")}


def validate_config(
    config: Mapping[str, Any], known_effects: tuple[str, ...], known_backends: tuple[str, ...]
) -> None:
    """Check the configuration for missing keys and invalid values.

    Args:
        config: Configuration mapping.
        known_effects: Effect names supported by the effect registry.
        known_backends: DSP backend names supported by the effect registry.

    Raises:
        ConfigError: Describing every problem found.
    """
    problems: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    try:
        n = get(config, "dataset.num_source_files")
        check(isinstance(n, int) and n >= 1, f"dataset.num_source_files must be a positive integer (got {n!r})")
        exts = get(config, "dataset.audio_extensions")
        check(bool(exts), "dataset.audio_extensions must not be empty")

        sr = get(config, "audio.sample_rate")
        check(isinstance(sr, int) and sr >= 8000, f"audio.sample_rate must be an integer >= 8000 (got {sr!r})")
        ch = get(config, "audio.channels")
        check(ch in (1, 2), f"audio.channels must be 1 or 2 (got {ch!r})")
        dur = get(config, "audio.duration")
        check(isinstance(dur, (int, float)) and dur > 0, f"audio.duration must be > 0 seconds (got {dur!r})")
        fmt = str(get(config, "audio.output_format")).lower()
        check(fmt == "wav", f"audio.output_format must be 'wav' (got {fmt!r})")
        subtype = get(config, "audio.output_subtype")
        check(subtype in VALID_SUBTYPES, f"audio.output_subtype must be one of {VALID_SUBTYPES} (got {subtype!r})")

        norm = get(config, "preprocessing.normalization")
        check(norm in VALID_NORMALIZATION, f"preprocessing.normalization must be one of {VALID_NORMALIZATION}")
        frac = get(config, "preprocessing.min_active_fraction")
        check(0.0 <= frac <= 1.0, "preprocessing.min_active_fraction must be in [0, 1]")
        check(get(config, "preprocessing.frame_length") > 0, "preprocessing.frame_length must be > 0")
        check(get(config, "preprocessing.window_hop_seconds") > 0, "preprocessing.window_hop_seconds must be > 0")

        mode = get(config, "generation.parameter_mode")
        check(mode in VALID_PARAMETER_MODES, f"generation.parameter_mode must be one of {VALID_PARAMETER_MODES}")
        check(isinstance(get(config, "generation.seed"), int), "generation.seed must be an integer")
        check(get(config, "generation.max_retries") >= 0, "generation.max_retries must be >= 0")

        ratios = {s: get(config, f"split.{s}") for s in SPLITS}
        check(all(r >= 0 for r in ratios.values()), "split ratios must be non-negative")
        check(math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-6),
              f"split ratios must sum to 1.0 (got {sum(ratios.values()):.4f})")
        check(ratios["train"] > 0, "split.train must be > 0")

        enabled = get(config, "effects.enabled")
        for name in enabled:
            check(name in known_effects, f"Unknown effect '{name}' in effects.enabled (known: {known_effects})")
        check(any(enabled.values()), "At least one effect must be enabled")
        backend = get(config, "effects.backend")
        check(backend in known_backends, f"effects.backend must be one of {known_backends} (got {backend!r})")

        params = get(config, "parameters")
        for effect in enabled_effects(config):
            if effect not in params:
                problems.append(f"parameters.{effect} is missing")
                continue
            for pname, spec in params[effect].items():
                _check_parameter_spec(f"parameters.{effect}.{pname}", spec, problems)
    except ConfigError as exc:
        problems.append(str(exc))

    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))


def _check_parameter_spec(key: str, spec: Mapping[str, Any], problems: list[str]) -> None:
    for field in ("fixed", "min", "max"):
        if field not in spec:
            problems.append(f"{key}.{field} is missing")
            return
    lo, hi, fixed = spec["min"], spec["max"], spec["fixed"]
    scale = spec.get("scale", "linear")
    if scale not in VALID_SCALES:
        problems.append(f"{key}.scale must be one of {VALID_SCALES}")
    if lo > hi:
        problems.append(f"{key}: min ({lo}) > max ({hi})")
    if scale == "log" and lo <= 0:
        problems.append(f"{key}: log scale requires min > 0")
    magnitude = abs(fixed) if spec.get("random_sign") else fixed
    if not lo <= magnitude <= hi:
        problems.append(f"{key}: fixed value {fixed} is outside [{lo}, {hi}]")
