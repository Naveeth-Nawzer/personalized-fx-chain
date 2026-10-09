from pathlib import Path

import numpy as np
import pyloudnorm
import pytest
import soundfile as sf

from src.data.discovery import describe_file
from src.data.preprocessing import PreprocessingSettings, SourceRejected, prepare_source
from src.utils.config import load_config, with_overrides
from tests.conftest import FAST_OVERRIDES, TEST_SR


@pytest.fixture
def settings() -> PreprocessingSettings:
    return PreprocessingSettings.from_config(with_overrides(load_config(), FAST_OVERRIDES))


def _write(tmp_path: Path, name: str, data: np.ndarray, sr: int) -> Path:
    path = tmp_path / "Song" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr)
    return path


def _noise(seconds: float, sr: int, channels: int = 1, level: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(1)
    data = level * rng.standard_normal((int(seconds * sr), channels))
    return data.squeeze() if channels == 1 else data


def test_resample_downmix_crop_and_normalize(tmp_path: Path, settings) -> None:
    path = _write(tmp_path, "vocals.wav", _noise(3.0, 22050, channels=2), 22050)
    prepared = prepare_source(describe_file(path, tmp_path), settings, seed=42)
    assert prepared.sample_rate == TEST_SR
    assert prepared.audio.shape == (1, TEST_SR)  # mono, 1 s
    rec = prepared.record
    assert rec.resampled and rec.original_sample_rate == 22050
    assert rec.channel_conversion == "2->1"
    assert rec.segment_end_s - rec.segment_start_s == pytest.approx(1.0)
    loudness = pyloudnorm.Meter(TEST_SR).integrated_loudness(prepared.audio[0].astype(np.float64))
    assert loudness == pytest.approx(settings.target_lufs, abs=0.1)


def test_same_seed_same_segment(tmp_path: Path, settings) -> None:
    path = _write(tmp_path, "drums.wav", _noise(5.0, TEST_SR), TEST_SR)
    info = describe_file(path, tmp_path)
    a = prepare_source(info, settings, seed=42)
    b = prepare_source(info, settings, seed=42)
    np.testing.assert_array_equal(a.audio, b.audio)
    assert a.record.resampled is False


def test_silent_file_rejected(tmp_path: Path, settings) -> None:
    path = _write(tmp_path, "silence.wav", np.zeros(3 * TEST_SR), TEST_SR)
    with pytest.raises(SourceRejected, match="non-silent"):
        prepare_source(describe_file(path, tmp_path), settings, seed=0)


def test_mostly_silent_file_uses_active_region(tmp_path: Path, settings) -> None:
    data = np.zeros(4 * TEST_SR)
    data[3 * TEST_SR :] = _noise(1.0, TEST_SR)  # only the last second is active
    path = _write(tmp_path, "vocals.wav", data, TEST_SR)
    prepared = prepare_source(describe_file(path, tmp_path), settings, seed=0)
    assert prepared.record.segment_start_s >= 2.5
    assert prepared.record.active_fraction >= settings.min_active_fraction


def test_short_file_rejected(tmp_path: Path, settings) -> None:
    path = _write(tmp_path, "bass.wav", _noise(0.5, TEST_SR), TEST_SR)
    with pytest.raises(SourceRejected, match="shorter"):
        prepare_source(describe_file(path, tmp_path), settings, seed=0)


def test_corrupt_file_rejected(tmp_path: Path, settings) -> None:
    path = tmp_path / "Song" / "broken.wav"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"RIFF....not really a wav")
    with pytest.raises(SourceRejected, match="cannot decode"):
        prepare_source(describe_file(path, tmp_path), settings, seed=0)


def test_peak_ceiling_respected(tmp_path: Path) -> None:
    # A very quiet, spiky signal would need a huge LUFS gain; the peak ceiling must cap it.
    config = with_overrides(load_config(), {**FAST_OVERRIDES, "preprocessing.target_lufs": -5.0})
    settings = PreprocessingSettings.from_config(config)
    data = np.zeros(2 * TEST_SR)
    data[::400] = 0.5
    path = _write(tmp_path, "clicks.wav", data, TEST_SR)
    prepared = prepare_source(describe_file(path, tmp_path), settings, seed=0)
    assert np.max(np.abs(prepared.audio)) <= 10 ** (settings.peak_ceiling_dbfs / 20) + 1e-6
