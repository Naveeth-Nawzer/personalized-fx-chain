import json
from pathlib import Path

from src.data.metadata import (
    CSV_COLUMNS,
    SUMMARY_FILE,
    read_split_csvs,
    read_table,
    record_to_row,
    row_to_record,
    write_table,
)

RECORD = {
    "sample_id": "sample_000001",
    "source_id": "src_0001",
    "source_file": "A Classic Education - NightOwl/vocals.wav",
    "source_song": "A Classic Education - NightOwl",
    "source_category": "vocals",
    "split": "train",
    "dry_path": "train/dry/sample_000001.wav",
    "wet_path": "train/wet/sample_000001.wav",
    "effect_chain": ["EQ", "Compressor"],
    "processing_order": ["EQ", "Compressor"],
    "combination": "EQ+Compressor",
    "label": [1, 1, 0],
    "eq_label": 1,
    "compressor_label": 1,
    "reverb_label": 0,
    "parameters": {"EQ": {"frequency": 1000.0, "gain": 6.0, "q": 1.0},
                   "Compressor": {"threshold": -24.0, "ratio": 4.0, "attack": 10.0, "release": 100.0,
                                  "makeup_gain": 3.0}},
    "parameters_normalized": {"EQ": {"frequency": 0.5, "gain": 0.75, "q": 0.33}},
    "effect_change_db": {"EQ": -7.0, "Compressor": -11.0},
    "low_effect_change": [],
    "sample_rate": 44100,
    "num_channels": 1,
    "duration_s": 10.0,
    "wet_peak_dbfs": -3.2,
    "wet_clipped": 0,
    "dsp_backend": "scipy",
    "parameter_mode": "fixed",
    "random_seed": 42,
    "sample_seed": 123456,
    "preprocessing": {"resampled": False, "segment_start_s": 12.5},
}


def test_row_round_trip() -> None:
    row = record_to_row(RECORD)
    assert set(row) == set(CSV_COLUMNS)
    assert json.loads(row["parameters"])["EQ"]["gain"] == 6.0
    assert row["label"] == "[1,1,0]"
    assert row_to_record(row) == RECORD


def test_csv_round_trip(tmp_path: Path) -> None:
    path = write_table(tmp_path / "train.csv", [RECORD], CSV_COLUMNS)
    assert read_table(path) == [RECORD]


def test_generated_metadata_files(generated_dataset) -> None:
    meta = generated_dataset.metadata_dir
    for name in ("train.csv", "validation.csv", "test.csv", SUMMARY_FILE, "selected_sources.csv",
                 "skipped_files.csv", "failures.csv", "config_used.yaml"):
        assert (meta / name).is_file(), name

    by_split = read_split_csvs(meta)
    rows = [r for split_rows in by_split.values() for r in split_rows]
    assert len(rows) == 42
    for r in rows:
        assert set(CSV_COLUMNS) <= set(r)
        assert sorted(r["parameters"]) == sorted(r["effect_chain"])
        assert r["random_seed"] == 42
        # Source category is metadata only: it never appears in the label.
        assert r["label"] == [r["eq_label"], r["compressor_label"], r["reverb_label"]]


def test_summary_contents(generated_dataset) -> None:
    summary = json.loads((generated_dataset.metadata_dir / SUMMARY_FILE).read_text())
    assert summary["num_source_files"] == 6
    assert summary["num_generated_pairs"] == 42
    assert summary["num_train_pairs"] + summary["num_validation_pairs"] + summary["num_test_pairs"] == 42
    assert summary["effect_distribution"] == {"EQ": 24, "Compressor": 24, "Reverb": 24}
    assert set(summary["combination_distribution"].values()) == {6}
    assert summary["label_names"] == ["EQ", "Compressor", "Reverb"]
    assert summary["random_seed"] == 42
    assert summary["parameter_mode"] == "fixed"
    assert summary["audio"]["sample_rate"] == 16000
    assert "numpy" in summary["software_versions"]
    assert summary["generation_timestamp_utc"]
