"""
tests/test_rl_env.py
====================
Unit tests for the Goal-Conditioned Audio Effects RL Environment (AudioFXEnv).
"""

import math
from pathlib import Path
import pytest
import torch
from rl.dapn_env import AudioFXEnv


@pytest.fixture
def dummy_audio():
    """Generates synthetic 2-second audio pair."""
    sr = 44100
    t = torch.linspace(0, 2.0, sr * 2)
    # Dry: pure tone
    dry = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    # Target: tone + harmonics
    target = (torch.sin(2 * 3.14159 * 440 * t) + 0.5 * torch.sin(2 * 3.14159 * 880 * t)).unsqueeze(0)
    return dry, target


def test_env_init():
    """Test environment initialization and spaces."""
    env = AudioFXEnv(predictor_checkpoint=None)
    assert env.action_dim == 36
    assert env.obs_dim == 100
    assert env.num_effects == 7
    assert env.num_params == 29


def test_env_reset(dummy_audio):
    """Test reset without warm-start."""
    dry, target = dummy_audio
    env = AudioFXEnv(predictor_checkpoint=None)

    obs, info = env.reset(dry, target, use_warm_start=False)

    assert obs.shape == (100,)
    assert isinstance(info["initial_loss"], float)
    assert info["num_active_effects"] == 0
    assert env.current_step == 0


def test_env_reset_warm_start(dummy_audio):
    """Test reset with trained model checkpoint warm-start."""
    dry, target = dummy_audio
    ckpt_path = Path("checkpoints/best_model.pth")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint not found for warm-start test")

    env = AudioFXEnv(predictor_checkpoint=ckpt_path)
    obs, info = env.reset(dry, target, use_warm_start=True)

    assert obs.shape == (100,)
    assert info["warm_started"] is True
    # The predictor should have initialized candidate presence/params
    assert env.current_presence.shape == (7,)
    assert env.current_params.shape == (29,)


def test_env_step(dummy_audio):
    """Test step execution, action effects, and reward computation."""
    dry, target = dummy_audio
    env = AudioFXEnv(predictor_checkpoint=None, max_steps_per_episode=5)
    obs, _ = env.reset(dry, target, use_warm_start=False)

    # Action: Turn on Reverb (index 3 > 0), adjust first param
    action = torch.zeros(36)
    action[3] = 1.0  # Turn on Reverb
    action[7] = 0.5  # Adjust first parameter

    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs.shape == (100,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert info["step"] == 1
    assert "improvement_pct" in info
    assert env.current_presence[3].item() == 1.0


def test_env_truncation(dummy_audio):
    """Test episode truncation after max steps."""
    dry, target = dummy_audio
    env = AudioFXEnv(predictor_checkpoint=None, max_steps_per_episode=3)
    env.reset(dry, target, use_warm_start=False)

    action = torch.zeros(36)
    for _ in range(2):
        _, _, _, truncated, _ = env.step(action)
        assert truncated is False

    _, _, _, truncated, _ = env.step(action)
    assert truncated is True


def test_env_random_actions_finite_observations_and_rewards(dummy_audio):
    """Verify that a sequence of arbitrary continuous actions produces finite observations and rewards."""
    dry, target = dummy_audio
    env = AudioFXEnv(predictor_checkpoint=None, max_steps_per_episode=10)
    obs, info = env.reset(dry, target, use_warm_start=False)
    assert torch.isfinite(obs).all()
    assert math.isfinite(info["current_loss"])

    torch.manual_seed(123)
    for _ in range(8):
        action = torch.randn(36)
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        assert next_obs.shape == (100,)
        assert torch.isfinite(next_obs).all(), "Observation contains non-finite values"
        assert math.isfinite(reward), f"Reward {reward} is not finite"
        assert math.isfinite(step_info["current_loss"])
        # Parameters must remain strictly in [0, 1]
        assert (env.current_params >= 0.0).all() and (env.current_params <= 1.0).all()
        # Presence must be 0 or 1
        assert torch.all((env.current_presence == 0.0) | (env.current_presence == 1.0))
        if terminated or truncated:
            break


def test_env_extreme_actions_clamping(dummy_audio):
    """Extreme actions (+100, -100) are clamped gracefully without NaN."""
    dry, target = dummy_audio
    env = AudioFXEnv(predictor_checkpoint=None)
    env.reset(dry, target, use_warm_start=False)

    huge_positive_action = torch.full((36,), 100.0)
    obs, reward, _, _, _ = env.step(huge_positive_action)
    assert torch.isfinite(obs).all()
    assert math.isfinite(reward)
    assert torch.allclose(env.current_params, torch.ones_like(env.current_params))

    huge_negative_action = torch.full((36,), -100.0)
    obs, reward, _, _, _ = env.step(huge_negative_action)
    assert torch.isfinite(obs).all()
    assert math.isfinite(reward)
    assert torch.allclose(env.current_params, torch.zeros_like(env.current_params))
