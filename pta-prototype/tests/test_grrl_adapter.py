"""Unit tests for GRRLAdapter and taste projection logic.

Tests the interface boundary without requiring a GPU, PyTorch, DAPN checkpoint,
or actual RL model training.
"""
import numpy as np
import pytest

from pta.core.preference_mapping import PREFERENCE_DIMENSIONS
from pta.integration.grrl_adapter import (
    GRRLAdapter,
    project_taste,
    validate_constraints,
    validate_preferences,
)


# --- Test doubles (Fakes) ---

class FakeGRRLEnvironment:
    """Lightweight test double simulating the GRRL environment."""

    def __init__(self, fail_on_run=False):
        self.fail_on_run = fail_on_run
        self.received_taste = None
        self.received_constraints = None
        self.current_chain = []
        self.steps = 0
        self.max_steps = 4

    def reset(self, dry_audio=None, target_audio=None, pta_taste=None, constraints=None):
        self.received_taste = pta_taste
        self.received_constraints = constraints or {}
        self.current_chain = []
        self.steps = 0
        return np.zeros(32, dtype=np.float32)

    def sample_action(self):
        # Honor prohibited effects mask
        prohibited = set(self.received_constraints.get("prohibited_effects", []))
        candidate_effects = ["EQ", "Compressor", "Delay", "Reverb", "Distortion"]
        allowed = [e for e in candidate_effects if e not in prohibited]
        chosen = allowed[0] if allowed else "EQ"
        return chosen

    def step(self, action):
        if self.fail_on_run:
            raise RuntimeError("DSP rendering crashed in DAPN")

        self.steps += 1
        effect_name = action if isinstance(action, str) else "EQ"
        
        # Apply parameter bounds if defined
        bounds = self.received_constraints.get("parameter_bounds", {}).get(effect_name, {})
        params = {"gain": 0.5, "wet_mix": 0.5}
        for p, interval in bounds.items():
            params[p] = (interval[0] + interval[1]) / 2.0

        self.current_chain.append({"effect": effect_name, "parameters": params})
        done = self.steps >= 2  # stop after 2 steps
        info = {
            "chain": self.current_chain,
            "metrics": {"loss": 0.12, "iterations": self.steps},
            "audio_ref": "outputs/audio/rendered_fake.wav",
        }
        return np.zeros(32, dtype=np.float32), 1.0, done, info


# --- Tests for Preference Validation ---

def test_validate_preferences_valid_dict():
    prefs = {dim: 0.0 for dim in PREFERENCE_DIMENSIONS}
    prefs["reverb"] = 0.5
    v = validate_preferences(prefs)
    assert v.shape == (7,)
    assert v[2] == 0.5


def test_validate_preferences_valid_array():
    arr = np.array([0.1, -0.2, 0.3, 0.0, 0.0, 0.5, -0.1], dtype=np.float32)
    v = validate_preferences(arr)
    assert np.allclose(v, arr)


def test_validate_preferences_missing_dimension():
    incomplete = {"brightness": 0.0, "reverb": 0.5}
    with pytest.raises(ValueError, match="Missing required preference dimensions"):
        validate_preferences(incomplete)


def test_validate_preferences_out_of_range():
    prefs = {dim: 0.0 for dim in PREFERENCE_DIMENSIONS}
    prefs["reverb"] = 1.5
    with pytest.raises(ValueError, match=r"All preference values must be in \[-1.0, 1.0\]"):
        validate_preferences(prefs)


def test_validate_preferences_nan_rejected():
    prefs = {dim: 0.0 for dim in PREFERENCE_DIMENSIONS}
    prefs["compression"] = float("nan")
    with pytest.raises(ValueError, match="finite numbers"):
        validate_preferences(prefs)


def test_validate_preferences_inf_rejected():
    prefs = {dim: 0.0 for dim in PREFERENCE_DIMENSIONS}
    prefs["delay"] = float("inf")
    with pytest.raises(ValueError, match="finite numbers"):
        validate_preferences(prefs)


# --- Tests for 16-Dimensional Taste Projection ---

def test_project_taste_shape_and_finite():
    prefs = {dim: 0.0 for dim in PREFERENCE_DIMENSIONS}
    prefs["reverb"] = 0.6
    prefs["compression"] = -0.3
    z = project_taste(prefs)
    assert z.shape == (16,)
    assert np.isfinite(z).all()
    assert np.all(z >= -1.0) and np.all(z <= 1.0)


def test_project_taste_neutral_baseline():
    neutral = {dim: 0.0 for dim in PREFERENCE_DIMENSIONS}
    z = project_taste(neutral)
    # Neutral input should produce all zeros across raw, interactions, and aggregates
    assert np.allclose(z, 0.0)


def test_project_taste_preserves_first_seven():
    raw = [0.1, -0.4, 0.7, 0.0, 0.2, -0.3, 0.5]
    prefs = dict(zip(PREFERENCE_DIMENSIONS, raw))
    z = project_taste(prefs)
    assert np.allclose(z[:7], raw, atol=1e-5)


def test_project_taste_computes_interactions():
    prefs = {dim: 0.0 for dim in PREFERENCE_DIMENSIONS}
    prefs["compression"] = 0.5  # index 1
    prefs["reverb"] = 0.4       # index 2
    z = project_taste(prefs)
    # z[7] is compression x reverb = 0.5 * 0.4 = 0.20
    assert np.isclose(z[7], 0.20, atol=1e-5)


# --- Tests for Constraints Validation ---

def test_validate_constraints_valid():
    raw_c = {
        "prohibited_effects": ["Distortion", "Flanger"],
        "parameter_bounds": {
            "Reverb": {"wet_mix": [0.3, 0.8]},
            "Compressor": {"ratio": [0.1, 0.5]},
        },
    }
    vc = validate_constraints(raw_c)
    assert "Distortion" in vc["prohibited_effects"]
    assert vc["parameter_bounds"]["Reverb"]["wet_mix"] == [0.3, 0.8]


def test_validate_constraints_unknown_effect():
    with pytest.raises(ValueError, match="Unknown effect"):
        validate_constraints({"prohibited_effects": ["Phaser_Unsupported"]})


def test_validate_constraints_invalid_interval():
    bad_interval = {
        "parameter_bounds": {"Reverb": {"wet_mix": [0.9, 0.2]}}  # low > high
    }
    with pytest.raises(ValueError, match="Invalid bound range"):
        validate_constraints(bad_interval)


# --- Tests for GRRLAdapter Execution ---

def test_adapter_generates_successfully():
    env = FakeGRRLEnvironment()
    adapter = GRRLAdapter(env_factory=lambda: env)
    
    prefs = {dim: 0.0 for dim in PREFERENCE_DIMENSIONS}
    prefs["reverb"] = 0.5
    constraints = {
        "prohibited_effects": ["Distortion"],
        "parameter_bounds": {"Reverb": {"wet_mix": [0.4, 0.9]}},
    }

    res = adapter.generate(
        dry_audio="dry.wav",
        target_audio="ref.wav",
        preferences=prefs,
        constraints=constraints,
        session_id="S001",
    )

    assert res["status"] == "success"
    assert res["session_id"] == "S001"
    assert len(res["chain"]) > 0
    # Prohibited effect was not selected
    assert all(e["effect"] != "Distortion" for e in res["chain"])
    # Taste vector was delivered
    assert env.received_taste is not None
    assert env.received_taste.shape == (16,)
    assert "audio_ref" in res


def test_adapter_generate_initial():
    env = FakeGRRLEnvironment()
    adapter = GRRLAdapter(env_factory=lambda: env)

    res = adapter.generate_initial("dry.wav", "ref.wav", session_id="S001")
    assert res["status"] == "success"
    # Applied taste was neutral
    assert np.allclose(res["applied_taste"], 0.0)


def test_adapter_missing_env_factory_fails_gracefully():
    adapter = GRRLAdapter(env_factory=None)
    res = adapter.generate("dry.wav", "ref.wav", {dim: 0.0 for dim in PREFERENCE_DIMENSIONS})
    assert res["status"] == "failed"
    assert res["error_type"] == "ConfigurationError"


def test_adapter_runtime_failure_handled_gracefully():
    failing_env = FakeGRRLEnvironment(fail_on_run=True)
    adapter = GRRLAdapter(env_factory=lambda: failing_env)

    res = adapter.generate("dry.wav", "ref.wav", {dim: 0.0 for dim in PREFERENCE_DIMENSIONS})
    assert res["status"] == "failed"
    assert res["error_type"] == "ExecutionError"
    assert "crashed in DAPN" in res["error_message"]
