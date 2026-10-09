"""Recursive discovery of source audio files.

Works with MUSDB-style layouts (``<root>/<song>/<stem>.wav``) and any other
nesting. Nothing is hard-coded to particular stem names:

* ``song``     = the file's parent directory relative to the root (POSIX form),
                 or the file stem when the file sits directly in the root.
                 Used to group files for leakage-free splitting.
* ``category`` = the filename stem, lower-cased (e.g. ``vocals``). Metadata only;
                 it is never a prediction target.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioFileInfo:
    """A discovered source audio file."""

    path: Path
    rel_path: str
    song: str
    category: str
    extension: str


def discover_audio_files(
    root: str | Path,
    extensions: Iterable[str] = (".wav", ".flac", ".mp3"),
    include_categories: Sequence[str] | None = None,
    exclude_categories: Sequence[str] = (),
) -> list[AudioFileInfo]:
    """Find audio files under ``root`` recursively.

    Args:
        root: Dataset root directory (read only).
        extensions: Accepted file extensions (case-insensitive).
        include_categories: If given, keep only these categories.
        exclude_categories: Categories to drop.

    Returns:
        Files sorted by relative path (stable across operating systems).
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {root}")
    exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    include = {c.lower() for c in include_categories} if include_categories else None
    exclude = {c.lower() for c in exclude_categories}

    files: list[AudioFileInfo] = []
    filtered = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in exts:
            continue
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue  # hidden files/folders (e.g. ._macOS resource forks)
        info = describe_file(path, root)
        if (include is not None and info.category not in include) or info.category in exclude:
            filtered += 1
            continue
        files.append(info)

    files.sort(key=lambda f: f.rel_path)
    if filtered:
        logger.info("Excluded %d file(s) by category filter", filtered)
    return files


def describe_file(path: Path, root: Path) -> AudioFileInfo:
    """Build an :class:`AudioFileInfo` for ``path`` relative to ``root``."""
    rel = path.relative_to(root)
    parent = rel.parent.as_posix()
    song = parent if parent not in ("", ".") else path.stem
    return AudioFileInfo(
        path=path,
        rel_path=rel.as_posix(),
        song=song,
        category=path.stem.lower(),
        extension=path.suffix.lower(),
    )
