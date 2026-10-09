from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data.discovery import discover_audio_files


def _write(path: Path, sr: int = 8000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fmt = "FLAC" if path.suffix.lower() == ".flac" else "WAV"
    sf.write(str(path), np.zeros(sr, dtype=np.float32), sr, format=fmt)


def test_recursive_discovery_with_arbitrary_names(tmp_path: Path) -> None:
    _write(tmp_path / "Song A" / "vocals.wav")
    _write(tmp_path / "Song A" / "Lead Guitar.flac")
    _write(tmp_path / "nested" / "deeper" / "Song B" / "drums.WAV")
    _write(tmp_path / "loose_take.wav")
    (tmp_path / "Song A" / "notes.txt").write_text("not audio")
    (tmp_path / "Song A" / "cover.png").write_bytes(b"\x89PNG")

    files = discover_audio_files(tmp_path)
    rel = [f.rel_path for f in files]
    assert rel == sorted(rel)
    assert set(rel) == {"Song A/vocals.wav", "Song A/Lead Guitar.flac",
                        "nested/deeper/Song B/drums.WAV", "loose_take.wav"}

    by_rel = {f.rel_path: f for f in files}
    assert by_rel["Song A/Lead Guitar.flac"].category == "lead guitar"
    assert by_rel["Song A/Lead Guitar.flac"].song == "Song A"
    assert by_rel["nested/deeper/Song B/drums.WAV"].song == "nested/deeper/Song B"
    assert by_rel["nested/deeper/Song B/drums.WAV"].category == "drums"
    # A file directly in the root is its own group.
    assert by_rel["loose_take.wav"].song == "loose_take"


def test_extension_and_category_filters(tmp_path: Path) -> None:
    for stem in ("bass", "drums", "mixture", "vocals"):
        _write(tmp_path / "S1" / f"{stem}.wav")
    _write(tmp_path / "S1" / "other.flac")

    only_wav = discover_audio_files(tmp_path, extensions=[".wav"])
    assert {f.category for f in only_wav} == {"bass", "drums", "mixture", "vocals"}

    no_mix = discover_audio_files(tmp_path, exclude_categories=["mixture"])
    assert "mixture" not in {f.category for f in no_mix}

    include = discover_audio_files(tmp_path, include_categories=["Vocals", "bass"])
    assert {f.category for f in include} == {"vocals", "bass"}


def test_hidden_files_are_ignored(tmp_path: Path) -> None:
    _write(tmp_path / "S1" / "vocals.wav")
    _write(tmp_path / "S1" / "._vocals.wav")
    _write(tmp_path / ".cache" / "x.wav")
    assert [f.rel_path for f in discover_audio_files(tmp_path)] == ["S1/vocals.wav"]


def test_toy_dataset_discovery(toy_dataset: Path) -> None:
    files = discover_audio_files(toy_dataset)
    assert len(files) == 40
    assert {f.category for f in files} == {"bass", "drums", "other", "vocals", "mixture"}
    assert len({f.song for f in files}) == 8


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_audio_files(tmp_path / "does-not-exist")
