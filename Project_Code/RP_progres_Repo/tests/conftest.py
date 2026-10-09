"""Shared fixtures: tiny synthetic datasets and a fast test configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from src.data.generator import DatasetGenerator, GenerationReport
from src.utils.config import load_config, with_overrides
from src.utils.synthetic import make_toy_dataset

TEST_SR = 16000

#: Settings that keep tests fast (1 s clips at 16 kHz).
FAST_OVERRIDES: dict[str, Any] = {
    "audio.sample_rate": TEST_SR,
    "audio.duration": 1.0,
    "preprocessing.frame_length": 512,
    "preprocessing.window_hop_seconds": 0.25,
    "dataset.num_source_files": 6,
}


@pytest.fixture(scope="session")
def toy_dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """8 songs x 5 stems (bass, drums, other, vocals, mixture), 3 s, 22.05 kHz stereo."""
    root = tmp_path_factory.mktemp("toy_dataset")
    make_toy_dataset(root, num_songs=8, duration=3.0, sample_rate=22050, channels=2, seed=0)
    return root


@pytest.fixture
def make_config() -> Callable[..., dict[str, Any]]:
    """Factory: default config + fast test settings + optional dotted-key overrides."""

    def _make(**overrides: Any) -> dict[str, Any]:
        merged = dict(FAST_OVERRIDES)
        merged.update({k.replace("__", "."): v for k, v in overrides.items()})
        return with_overrides(load_config(), merged)

    return _make


def run_generation(config: dict[str, Any], input_dir: Path, base: Path, overwrite: bool = False) -> GenerationReport:
    return DatasetGenerator(config).run(input_dir, base / "out", base / "meta", overwrite=overwrite)


@pytest.fixture(scope="session")
def generated_dataset(toy_dataset: Path, tmp_path_factory: pytest.TempPathFactory) -> GenerationReport:
    """A generated 6-source (42-pair) dataset shared by read-only tests."""
    from src.utils.config import load_config as _load

    config = with_overrides(_load(), FAST_OVERRIDES)
    return run_generation(config, toy_dataset, tmp_path_factory.mktemp("generated"))


@pytest.fixture
def tone() -> np.ndarray:
    """1 s, 16 kHz mono test signal: two tones with an amplitude envelope plus a little noise."""
    rng = np.random.default_rng(0)
    t = np.arange(TEST_SR) / TEST_SR
    env = 0.5 * (1 + np.sin(2 * np.pi * 3 * t))
    x = 0.4 * env * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 1500 * t))
    x += 0.01 * rng.standard_normal(t.size)
    return x.astype(np.float32)[np.newaxis, :]
