"""Each test corrupts a copy of a generated dataset and checks the validator catches it."""

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.data.validation import validate_dataset


@pytest.fixture
def dataset_copy(generated_dataset, tmp_path: Path) -> tuple[Path, Path]:
    out = shutil.copytree(generated_dataset.output_dir, tmp_path / "out")
    meta = shutil.copytree(generated_dataset.metadata_dir, tmp_path / "meta")
    return Path(out), Path(meta)


def _edit_csv(path: Path, edit) -> None:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields, rows = reader.fieldnames, list(reader)
    edit(rows)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_clean_dataset_passes(generated_dataset) -> None:
    report = validate_dataset(generated_dataset.output_dir, generated_dataset.metadata_dir)
    assert report.ok, report.format()
    assert report.num_samples == 42


def test_missing_wet_file(dataset_copy) -> None:
    out, meta = dataset_copy
    next((out / "train" / "wet").glob("*.wav")).unlink()
    report = validate_dataset(out, meta)
    assert not report.ok and report.has_issue("paths")


def test_corrupt_audio_file(dataset_copy) -> None:
    out, meta = dataset_copy
    next((out / "test" / "dry").glob("*.wav")).write_bytes(b"garbage")
    report = validate_dataset(out, meta)
    assert report.has_issue("audio_files")


def test_length_and_rate_mismatch(dataset_copy) -> None:
    out, meta = dataset_copy
    wet = next((out / "validation" / "wet").glob("*.wav"))
    sf.write(str(wet), np.zeros(8000, dtype=np.float32), 8000, subtype="FLOAT")
    report = validate_dataset(out, meta)
    assert report.has_issue("audio_compatibility")


def test_wrong_label(dataset_copy) -> None:
    out, meta = dataset_copy

    def flip(rows):
        rows[0]["reverb_label"] = "1" if rows[0]["reverb_label"] == "0" else "0"

    _edit_csv(meta / "train.csv", flip)
    assert validate_dataset(out, meta, check_audio=False).has_issue("labels")


def test_all_zero_label(dataset_copy) -> None:
    out, meta = dataset_copy

    def zero(rows):
        rows[0].update(label="[0,0,0]", eq_label="0", compressor_label="0", reverb_label="0")

    _edit_csv(meta / "train.csv", zero)
    report = validate_dataset(out, meta, check_audio=False)
    assert any("no effect is active" in i.message for i in report.errors)


def test_wrong_label_length(dataset_copy) -> None:
    out, meta = dataset_copy
    _edit_csv(meta / "train.csv", lambda rows: rows[0].update(label="[1,0]"))
    report = validate_dataset(out, meta, check_audio=False)
    assert any("exactly 3 values" in i.message for i in report.errors)


def test_non_canonical_processing_order(dataset_copy) -> None:
    out, meta = dataset_copy

    def reorder(rows):
        for r in rows:
            if r["combination"] == "EQ+Compressor":
                r["processing_order"] = '["Compressor","EQ"]'
                return

    _edit_csv(meta / "train.csv", reorder)
    assert validate_dataset(out, meta, check_audio=False).has_issue("effect_chain")


def test_duplicate_sample_id(dataset_copy) -> None:
    out, meta = dataset_copy
    _edit_csv(meta / "train.csv", lambda rows: rows[1].update(sample_id=rows[0]["sample_id"]))
    assert validate_dataset(out, meta, check_audio=False).has_issue("sample_ids")


def test_source_leakage_between_splits(dataset_copy) -> None:
    out, meta = dataset_copy
    with (meta / "train.csv").open(newline="", encoding="utf-8") as fh:
        train_song = next(csv.DictReader(fh))["source_song"]
    _edit_csv(meta / "test.csv", lambda rows: rows[0].update(source_song=train_song))
    report = validate_dataset(out, meta, check_audio=False)
    assert report.has_issue("source_leakage")


def test_missing_combination(dataset_copy) -> None:
    out, meta = dataset_copy
    _edit_csv(meta / "test.csv", lambda rows: rows.pop())
    report = validate_dataset(out, meta, check_audio=False)
    assert report.has_issue("completeness")
    assert report.has_issue("summary_consistency")


def test_path_outside_root(dataset_copy) -> None:
    out, meta = dataset_copy
    _edit_csv(meta / "train.csv", lambda rows: rows[0].update(dry_path="../../etc/passwd"))
    report = validate_dataset(out, meta, check_audio=False)
    assert report.has_issue("paths") and report.has_issue("split_consistency")


def test_missing_metadata(tmp_path: Path) -> None:
    report = validate_dataset(tmp_path / "out", tmp_path / "meta")
    assert not report.ok and report.has_issue("metadata_files")


def test_report_format(dataset_copy) -> None:
    out, meta = dataset_copy
    summary = json.loads((meta / "dataset_summary.json").read_text())
    summary["num_generated_pairs"] = 1
    (meta / "dataset_summary.json").write_text(json.dumps(summary))
    text = validate_dataset(out, meta, check_audio=False).format()
    assert "[FAIL] summary_consistency" in text and "RESULT: FAILED" in text
