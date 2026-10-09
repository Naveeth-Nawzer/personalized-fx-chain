import numpy as np
import pytest

from src.effects.parameters import ParameterSampler, ParameterSpec
from src.utils.config import ConfigError, load_config, validate_config, with_overrides
from src.effects import BACKENDS, CANONICAL_ORDER


@pytest.fixture
def config():
    return load_config()


def test_default_config_is_valid(config) -> None:
    validate_config(config, CANONICAL_ORDER, tuple(BACKENDS))


def test_fixed_mode_returns_config_values(config) -> None:
    sampler = ParameterSampler.from_config(config["parameters"], "fixed")
    for effect in CANONICAL_ORDER:
        params = sampler.sample(effect)
        assert params == {k: v["fixed"] for k, v in config["parameters"][effect].items()}


def test_random_mode_within_ranges_and_reproducible(config) -> None:
    sampler = ParameterSampler.from_config(config["parameters"], "random")
    for effect in CANONICAL_ORDER:
        draws_a = [sampler.sample(effect, np.random.default_rng(i)) for i in range(200)]
        draws_b = [sampler.sample(effect, np.random.default_rng(i)) for i in range(200)]
        assert draws_a == draws_b
        for draw in draws_a:
            for name, value in draw.items():
                spec = config["parameters"][effect][name]
                magnitude = abs(value) if spec.get("random_sign") else value
                assert spec["min"] - 1e-9 <= magnitude <= spec["max"] + 1e-9


def test_random_sign_produces_boosts_and_cuts(config) -> None:
    sampler = ParameterSampler.from_config(config["parameters"], "random")
    gains = [sampler.sample("EQ", np.random.default_rng(i))["gain"] for i in range(100)]
    assert any(g > 0 for g in gains) and any(g < 0 for g in gains)
    assert all(abs(g) >= 3.0 for g in gains)  # never an (inaudible) near-zero EQ gain


def test_random_mode_requires_rng(config) -> None:
    sampler = ParameterSampler.from_config(config["parameters"], "random")
    with pytest.raises(ValueError):
        sampler.sample("EQ")


def test_normalization_in_unit_interval(config) -> None:
    sampler = ParameterSampler.from_config(config["parameters"], "random")
    for i in range(50):
        for effect in CANONICAL_ORDER:
            params = sampler.sample(effect, np.random.default_rng(i))
            norm = sampler.normalize(effect, params)
            assert set(norm) == set(params)
            assert all(0.0 <= v <= 1.0 for v in norm.values())


def test_spec_normalization_examples() -> None:
    lin = ParameterSpec("x", fixed=5, min=0, max=10)
    assert lin.normalize(5) == 0.5
    log = ParameterSpec("f", fixed=1000, min=100, max=10000, scale="log")
    assert log.normalize(1000) == pytest.approx(0.5)
    signed = ParameterSpec("g", fixed=6, min=3, max=12, random_sign=True)
    assert signed.normalize(0.0) == 0.5 and signed.normalize(-12) == 0.0
    const = ParameterSpec("w", fixed=1, min=1, max=1, norm_range=(0.0, 1.0))
    assert const.normalize(1.0) == 1.0


def test_invalid_config_reports_problems(config) -> None:
    bad = with_overrides(config, {"split.train": 0.9, "generation.parameter_mode": "chaos",
                                  "parameters.EQ.q.min": -1.0, "parameters.EQ.q.scale": "log"})
    with pytest.raises(ConfigError) as info:
        validate_config(bad, CANONICAL_ORDER, tuple(BACKENDS))
    message = str(info.value)
    assert "sum to 1.0" in message and "parameter_mode" in message and "log scale" in message
