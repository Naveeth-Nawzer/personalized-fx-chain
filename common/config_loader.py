"""
common/config_loader.py
=======================
Centralised YAML configuration loader.

Functions
---------
load_system_config  — load config/system_config.yaml
load_effects_config — load config/effects_config.yaml

Both functions cache their result after the first call so subsequent
imports are free.  Override paths are supported for testing.
"""

from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Resolve the project root relative to this file's location
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SYSTEM_CONFIG  = _PROJECT_ROOT / "config" / "system_config.yaml"
_DEFAULT_EFFECTS_CONFIG = _PROJECT_ROOT / "config" / "effects_config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file.

    Parameters
    ----------
    path : Path
        Absolute path to the YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    yaml.YAMLError
        If the YAML file cannot be parsed.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Make sure you are running from the project root directory."
        )
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML file {path} to contain a mapping (dict), "
            f"got {type(data).__name__}."
        )
    logger.debug("Loaded config from %s", path)
    return data


@functools.lru_cache(maxsize=8)
def load_system_config(path: Path | None = None) -> dict[str, Any]:
    """Return the parsed system configuration dictionary.

    Parameters
    ----------
    path : Path, optional
        Override the default config/system_config.yaml path.
        Useful in tests that need isolated configs.

    Returns
    -------
    dict[str, Any]
        Parsed system config.
    """
    resolved = Path(path) if path is not None else _DEFAULT_SYSTEM_CONFIG
    return _load_yaml(resolved)


@functools.lru_cache(maxsize=8)
def load_effects_config(path: Path | None = None) -> dict[str, Any]:
    """Return the parsed effects configuration dictionary.

    Parameters
    ----------
    path : Path, optional
        Override the default config/effects_config.yaml path.
        Useful in tests that need isolated configs.

    Returns
    -------
    dict[str, Any]
        Parsed effects config.
    """
    resolved = Path(path) if path is not None else _DEFAULT_EFFECTS_CONFIG
    return _load_yaml(resolved)


def get_sample_rate() -> int:
    """Convenience accessor for the configured sample rate."""
    return int(load_system_config()["audio"]["sample_rate"])


def get_channels() -> int:
    """Convenience accessor for the configured number of channels."""
    return int(load_system_config()["audio"]["channels"])


def get_dtype() -> str:
    """Convenience accessor for the configured dtype string."""
    return str(load_system_config()["audio"]["dtype"])


def get_max_duration_seconds() -> float:
    """Convenience accessor for maximum allowed audio duration."""
    return float(load_system_config()["audio"]["max_duration_seconds"])
