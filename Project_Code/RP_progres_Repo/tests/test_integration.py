"""End-to-end tests on a tiny synthetic dataset (no real data needed)."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from src.cli.generate import main as generate_main
from src.cli.validate import main as validate_main
from src.data.generator import DatasetGenerator, GenerationError
from src.data.metadata import read_split_csvs
from src.data.validation import validate_dataset
from src.effects import EffectChain, EffectStep, create_effect
from src.utils.audio import load_audio
from tests.conftest import run_generation


def _all_rows(meta: Path) -> list[dict]:
    return [r for rows in read_split_csvs(meta).values() for r in rows]


def test_full_pipeline_produces_seven_pairs_per_source(generated_dataset) -> None:
    report = generated_dataset
    assert len(report.selection.selected) == 6
    assert len(report.records) == 42 and not report.failures
    per_source = Counter(r["source_id"] for r in report.records)
    assert set(per_source.values()) == {7}
    # All seven samples of a source share one split and one dry signal.
    for source in report.selection.selected:
        rows = [r for r in report.records if r["source_id"] == source.source_id]
        assert {r["split"] for r in rows} == {source.split}
        drys = [load_audio(report.output_dir / r["dry_path"])[0] for r in rows]
        assert all(np.array_equal(drys[0], d) for d in drys[1:])
    assert validate_dataset(report.output_dir, report.metadata_dir).ok


def test_wet_can_be_rerendered_from_metadata(generated_dataset) -> None:
    """Metadata fully describes the processing: re-rendering the dry with the stored parameters,
    in the stored processing order, reproduces the stored wet file."""
    out = generated_dataset.output_dir
    for row in _all_rows(generated_dataset.metadata_dir)[:14]:
        dry, sr = load_audio(out / row["dry_path"])
        wet, _ = load_audio(out / row["wet_path"])
        chain = EffectChain([EffectStep(create_effect(name, row["dsp_backend"]), row["parameters"][name])
                             for name in row["processing_order"]])
        np.testing.assert_allclose(chain.process(dry, sr), wet, atol=1e-6)


def test_same_seed_reproduces_metadata_and_audio(toy_dataset, make_config, tmp_path: Path) -> None:
    config = make_config(**{"dataset.num_source_files": 3, "generation.parameter_mode": "random"})
    a = run_generation(config, toy_dataset, tmp_path / "a")
    b = run_generation(config, toy_dataset, tmp_path / "b")
    rows_a, rows_b = _all_rows(a.metadata_dir), _all_rows(b.metadata_dir)
    assert rows_a == rows_b
    for row in rows_a:
        for col in ("dry_path", "wet_path"):
            # Compare decoded samples: float WAV headers carry a PEAK-chunk timestamp that differs per run.
            audio_a, sr_a = load_audio(a.output_dir / row[col])
            audio_b, sr_b = load_audio(b.output_dir / row[col])
            assert sr_a == sr_b
            np.testing.assert_array_equal(audio_a, audio_b)
    summary_a = json.loads((a.metadata_dir / "dataset_summary.json").read_text())
    summary_b = json.loads((b.metadata_dir / "dataset_summary.json").read_text())
    summary_a.pop("generation_timestamp_utc"), summary_b.pop("generation_timestamp_utc")
    assert summary_a == summary_b


def test_different_seed_changes_selection(toy_dataset, make_config, tmp_path: Path) -> None:
    a = run_generation(make_config(**{"generation.seed": 1}), toy_dataset, tmp_path / "a")
    b = run_generation(make_config(**{"generation.seed": 2}), toy_dataset, tmp_path / "b")
    files_a = [s.file.rel_path for s in a.selection.selected]
    files_b = [s.file.rel_path for s in b.selection.selected]
    assert files_a != files_b


def test_random_mode_parameters_vary_and_are_audible(toy_dataset, make_config, tmp_path: Path) -> None:
    report = run_generation(make_config(**{"generation.parameter_mode": "random"}), toy_dataset, tmp_path)
    gains = {r["parameters"]["EQ"]["gain"] for r in report.records if "EQ" in r["parameters"]}
    assert len(gains) > 5
    threshold = make_config()["generation"]["min_effect_change_db"]
    for r in report.records:
        assert all(v >= threshold for v in r["effect_change_db"].values())
    assert validate_dataset(report.output_dir, report.metadata_dir).ok


def test_effect_subset(toy_dataset, make_config, tmp_path: Path) -> None:
    config = make_config(**{"dataset.num_source_files": 3, "effects.enabled.Compressor": False})
    report = run_generation(config, toy_dataset, tmp_path)
    assert len(report.records) == 3 * 3
    assert {r["combination"] for r in report.records} == {"EQ", "Reverb", "EQ+Reverb"}
    assert all(r["compressor_label"] == 0 for r in report.records)
    assert validate_dataset(report.output_dir, report.metadata_dir).ok


def test_refuses_to_overwrite_without_flag(toy_dataset, make_config, tmp_path: Path) -> None:
    config = make_config(**{"dataset.num_source_files": 3})
    run_generation(config, toy_dataset, tmp_path)
    with pytest.raises(GenerationError, match="--overwrite"):
        run_generation(config, toy_dataset, tmp_path)
    report = run_generation(config, toy_dataset, tmp_path, overwrite=True)
    assert len(report.records) == 21


def test_refuses_output_inside_input(toy_dataset, make_config) -> None:
    with pytest.raises(GenerationError, match="inside the input dataset"):
        DatasetGenerator(make_config()).run(toy_dataset, toy_dataset / "generated", toy_dataset / "meta")


def test_input_dataset_not_modified(generated_dataset, toy_dataset) -> None:
    files = sorted(p.relative_to(toy_dataset).as_posix() for p in toy_dataset.rglob("*") if p.is_file())
    assert len(files) == 40 and all(f.endswith(".wav") for f in files)


def test_cli_smoke_test_and_validate(toy_dataset, tmp_path: Path) -> None:
    out, meta = tmp_path / "out", tmp_path / "meta"
    common = ["--input-dir", str(toy_dataset), "--output-dir", str(out), "--metadata-dir", str(meta),
              "--sample-rate", "16000", "--duration", "1.0"]
    assert generate_main([*common, "--smoke-test"]) == 0
    assert len(_all_rows(meta)) == 21  # 3 sources x 7 combinations
    assert (meta / "generation.log").is_file()
    assert validate_main(["--output-dir", str(out), "--metadata-dir", str(meta)]) == 0


def test_cli_reports_missing_input(tmp_path: Path) -> None:
    code = generate_main(["--input-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "o"),
                          "--metadata-dir", str(tmp_path / "m")])
    assert code == 1
