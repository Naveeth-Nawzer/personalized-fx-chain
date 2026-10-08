"""tests/test_config.py — verify configuration files load correctly."""
import pytest
from pathlib import Path
from common.config_loader import (
    load_system_config,
    load_effects_config,
    get_sample_rate,
    get_channels,
    get_dtype,
    get_max_duration_seconds,
)


def test_system_config_loads():
    cfg = load_system_config()
    assert isinstance(cfg, dict), "system_config should be a dict"


def test_system_config_audio_keys():
    cfg = load_system_config()
    audio = cfg["audio"]
    assert audio["sample_rate"] == 44100
    assert audio["channels"] == 1
    assert audio["audio_format"] == "WAV"
    assert audio["dtype"] == "float32"
    assert audio["max_duration_seconds"] == 10


def test_system_config_chain_keys():
    cfg = load_system_config()
    chain = cfg["chain"]
    assert chain["min_chain_length"] == 1
    assert chain["max_chain_length"] == 8
    assert chain["allow_repeated_effects"] is False


def test_effects_config_loads():
    cfg = load_effects_config()
    assert isinstance(cfg, dict)
    assert "effects" in cfg


def test_effects_config_has_all_required_effects():
    cfg = load_effects_config()
    effects = cfg["effects"]
    for name in ["EQ", "Compressor", "Reverb", "Distortion", "Delay", "Chorus", "Flanger", "STOP"]:
        assert name in effects, f"Missing effect: {name}"


def test_effects_config_eq_parameters():
    cfg = load_effects_config()
    eq = cfg["effects"]["EQ"]["parameters"]
    assert set(eq.keys()) == {"frequency", "gain", "q"}
    assert eq["frequency"]["physical_range"] == [20.0, 20000.0]
    assert eq["gain"]["physical_range"] == [-12.0, 12.0]
    assert eq["q"]["physical_range"] == [0.1, 10.0]


def test_convenience_accessors():
    assert get_sample_rate() == 44100
    assert get_channels() == 1
    assert get_dtype() == "float32"
    assert get_max_duration_seconds() == 10.0


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_system_config(Path("/nonexistent/path/system_config.yaml"))
