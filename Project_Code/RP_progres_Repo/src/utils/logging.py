"""Logging setup shared by the CLI entry points."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_HANDLER_TAG = "_inverse_fx_handler"


def setup_logging(level: str | int = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console (and optional file) logging for the ``src`` package.

    Calling this again replaces the handlers installed by a previous call,
    so it is safe to use from tests and repeated CLI invocations.
    """
    logger = logging.getLogger("src")
    logger.setLevel(level if isinstance(level, int) else level.upper())
    logger.propagate = False
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            handler.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    setattr(console, _HANDLER_TAG, True)
    logger.addHandler(console)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        setattr(file_handler, _HANDLER_TAG, True)
        logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (thin wrapper so modules don't depend on setup order)."""
    return logging.getLogger(name)
