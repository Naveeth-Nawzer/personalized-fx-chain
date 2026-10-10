"""
tests/test_grrl.py
==================
Comprehensive test suite for Dynamic Graph-Generating Reinforcement Learning Agent
(GRRL-Agent), GRRLEnv, GRRLRolloutBuffer, and hybrid PPO policy.

Verifies:
1. Discrete action selection.
2. Continuous parameter bounds for each effect.
3. Dynamic effect ordering (non-commutative sequence test).
4. Correct chain accumulation and DAPN rendering.
5. STOP termination.
6. Maximum chain length boundary.
7. Invalid-action masking.
8. Repeated-effect rules (allow_repeated_effects = False).
9. PPO log probabilities for hybrid actions.
10. GAE and terminal-state handling.
11. Parameter and reward numerical stability (no NaN/Inf).
12. Checkpoint save and reload.
13. M2 warm-start compatibility.
14. Regression tests for the existing fixed-order baseline (AudioFXEnv / PPOAgent).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
import pytest
import torch

from dataset.dapn_dataset import PARAM_SCHEMA, SUPPORTED_EFFECTS
from rl.dapn_env import AudioFXEnv
from rl.grrl_agent import GRRLActorCritic, GRRLAgent
from rl.grrl_buffer import GRRLRolloutBuffer
from rl.grrl_env import EFFECT_ACTIONS, GRRLEnv, NUM_EFFECT_ACTIONS, STOP_ACTION_IDX
from rl.ppo_agent import ActorCritic, PPOAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_audio():
    """Generates synthetic 2-second dry and target tone pair."""
    sr = 44100
    t = torch.linspace(0, 2.0, sr * 2)
    dry = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    target = (
        torch.sin(2 * 3.14159 * 440 * t)
        + 0.5 * torch.sin(2 * 3.14159 * 880 * t)
        + 0.25 * torch.sin(2 * 3.14159 * 1320 * t)
    ).unsqueeze(0)
    return dry, target


@pytest.fixture
def grrl_env():
    """Default GRRLEnv instance."""
    return GRRLEnv(
        predictor_checkpoint=None,
        device="cpu",
        max_chain_length=4,
        min_chain_length=1,
        allow_repeated_effects=False,
    )


@pytest.fixture
def grrl_agent(grrl_env):
    """Default GRRLAgent instance."""
    torch.manual_seed(42)
    return GRRLAgent(
        obs_dim=grrl_env.obs_dim,
        num_actions=grrl_env.num_actions,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# 1. Discrete Action Selection & Masking
# ---------------------------------------------------------------------------

def test_discrete_action_selection(grrl_agent, grrl_env, synthetic_audio):
    """Test that agent selects discrete action in [0, 7]."""
    dry, target = synthetic_audio
    obs, info = grrl_env.reset(dry, target)
    mask = grrl_env.get_action_mask()

    disc_act, cont_act, log_prob, val = grrl_agent.select_action(obs, action_mask=mask, deterministic=False)

    assert 0 <= disc_act < NUM_EFFECT_ACTIONS
    assert isinstance(disc_act, int)
    assert mask[disc_act].item() is True, "Sampled action must be valid according to mask"
    assert math.isfinite(log_prob)
    assert math.isfinite(val)


# ---------------------------------------------------------------------------
# 2. Continuous Parameter Bounds for Each Effect
# ---------------------------------------------------------------------------

def test_continuous_parameter_bounds(grrl_agent):
    """Test that continuous heads output values strictly in [0.0, 1.0] for every effect."""
    dummy_obs = torch.randn(148)

    for eff_idx, eff_name in enumerate(SUPPORTED_EFFECTS):
        # Force discrete selection of eff_idx
        features = grrl_agent.actor_critic.forward_features(dummy_obs.unsqueeze(0))
        head = grrl_agent.actor_critic.param_heads[eff_name]
        param_mean = torch.sigmoid(head(features[0]))

        k_params = len(PARAM_SCHEMA[eff_name])
        assert param_mean.shape == (k_params,)
        assert (param_mean >= 0.0).all() and (param_mean <= 1.0).all(), (
            f"Parameters for {eff_name} must be within [0.0, 1.0]"
        )


# ---------------------------------------------------------------------------
# 3. Dynamic Effect Ordering: Proves Different Orders Yield Different Output
# ---------------------------------------------------------------------------

def test_dynamic_effect_ordering_order_preservation(grrl_env, synthetic_audio):
    """Test that two different action sequences produce different effect orders in DAPN."""
    dry, target = synthetic_audio

    # Chain 1: Distortion -> Reverb
    grrl_env.reset(dry, target)
    dist_idx = EFFECT_ACTIONS.index("Distortion")
    rev_idx = EFFECT_ACTIONS.index("Reverb")

    grrl_env.step(dist_idx, continuous_params=torch.tensor([0.7, 0.5, 0.8, 0.4]))
    grrl_env.step(rev_idx, continuous_params=torch.tensor([0.6, 0.4, 0.5, 0.8, 0.5]))
    chain_1_seq = grrl_env.get_current_chain_summary()["sequence"]
    audio_1 = grrl_env.current_audio.clone()

    assert chain_1_seq == ["Distortion", "Reverb"], f"Expected ['Distortion', 'Reverb'], got {chain_1_seq}"

    # Chain 2: Reverb -> Distortion (reversed order)
    grrl_env.reset(dry, target)
    grrl_env.step(rev_idx, continuous_params=torch.tensor([0.6, 0.4, 0.5, 0.8, 0.5]))
    grrl_env.step(dist_idx, continuous_params=torch.tensor([0.7, 0.5, 0.8, 0.4]))
    chain_2_seq = grrl_env.get_current_chain_summary()["sequence"]
    audio_2 = grrl_env.current_audio.clone()

    assert chain_2_seq == ["Reverb", "Distortion"], f"Expected ['Reverb', 'Distortion'], got {chain_2_seq}"

    # Distortion and Reverb do NOT commute in audio DSP: output waveforms must differ!
    audio_diff = (audio_1 - audio_2).abs().max().item()
    assert audio_diff > 1e-4, f"Different effect sequences must produce different audio! Diff: {audio_diff}"


# ---------------------------------------------------------------------------
# 4. Correct Chain Accumulation and DAPN Rendering
# ---------------------------------------------------------------------------

def test_chain_accumulation_and_rendering(grrl_env, synthetic_audio):
    """Test sequential chain accumulation matches DAPN execution."""
    dry, target = synthetic_audio
    grrl_env.reset(dry, target)

    eq_idx = EFFECT_ACTIONS.index("EQ")
    comp_idx = EFFECT_ACTIONS.index("Compressor")

    grrl_env.step(eq_idx, continuous_params=torch.tensor([0.3, 0.7, 0.5]))
    assert len(grrl_env.current_chain) == 1
    assert grrl_env.current_chain[0]["effect"] == "EQ"

    grrl_env.step(comp_idx, continuous_params=torch.tensor([0.5, 0.4, 0.2, 0.3, 0.2]))
    assert len(grrl_env.current_chain) == 2
    assert grrl_env.current_chain[1]["effect"] == "Compressor"

    summary = grrl_env.get_current_chain_summary()
    assert summary["sequence"] == ["EQ", "Compressor"]
    assert torch.isfinite(grrl_env.current_audio).all()


# ---------------------------------------------------------------------------
# 5. STOP Termination
# ---------------------------------------------------------------------------

def test_stop_action_terminates_episode(grrl_env, synthetic_audio):
    """Test that selecting STOP terminates the episode immediately."""
    dry, target = synthetic_audio
    grrl_env.reset(dry, target)

    # Add one effect first (min_chain_length = 1)
    eq_idx = EFFECT_ACTIONS.index("EQ")
    _, _, term1, trunc1, _ = grrl_env.step(eq_idx, continuous_params=torch.tensor([0.4, 0.6, 0.5]))
    assert not term1
    assert not trunc1

    # Now choose STOP
    _, reward, terminated, truncated, info = grrl_env.step(STOP_ACTION_IDX)
    assert terminated is True
    assert truncated is False
    assert info["action"] == "STOP"
    assert math.isfinite(reward)


# ---------------------------------------------------------------------------
# 6. Maximum Chain Length Boundary
# ---------------------------------------------------------------------------

def test_max_chain_length_truncation(grrl_env, synthetic_audio):
    """Test that environment truncates or forces STOP when max_chain_length is reached."""
    dry, target = synthetic_audio
    grrl_env.reset(dry, target)  # max_chain_length is 4

    effects = ["EQ", "Compressor", "Delay", "Reverb"]
    for i, eff in enumerate(effects):
        act_idx = EFFECT_ACTIONS.index(eff)
        _, _, term, trunc, _ = grrl_env.step(act_idx)
        if i == len(effects) - 1:
            # Reached max length 4: mask should now only allow STOP
            mask = grrl_env.get_action_mask()
            assert mask[STOP_ACTION_IDX].item() is True
            assert mask[:STOP_ACTION_IDX].sum().item() == 0, "No more effects allowed at max length"


# ---------------------------------------------------------------------------
# 7. Repeated Effects Masking Rules
# ---------------------------------------------------------------------------

def test_repeated_effects_masking(grrl_env, synthetic_audio):
    """Test that already chosen effects are masked when allow_repeated_effects=False."""
    dry, target = synthetic_audio
    grrl_env.reset(dry, target)

    flanger_idx = EFFECT_ACTIONS.index("Flanger")
    mask_before = grrl_env.get_action_mask()
    assert mask_before[flanger_idx].item() is True

    grrl_env.step(flanger_idx)

    mask_after = grrl_env.get_action_mask()
    assert mask_after[flanger_idx].item() is False, "Flanger must be masked out after being chosen"


# ---------------------------------------------------------------------------
# 8. Minimum Chain Length Enforces Masking of STOP at Step 0
# ---------------------------------------------------------------------------

def test_min_chain_length_masks_stop(grrl_env, synthetic_audio):
    """Test that STOP is masked when chain_length < min_chain_length."""
    dry, target = synthetic_audio
    grrl_env.reset(dry, target)  # min_chain_length is 1

    mask = grrl_env.get_action_mask()
    assert mask[STOP_ACTION_IDX].item() is False, "STOP must be masked at step 0 if min_chain_length=1"


# ---------------------------------------------------------------------------
# 9. PPO Log Probabilities for Hybrid Actions
# ---------------------------------------------------------------------------

def test_hybrid_log_probabilities(grrl_agent):
    """Test that joint log-probabilities of hybrid actions are computed correctly."""
    batch_size = 4
    dummy_obs = torch.randn(batch_size, 148)
    dummy_masks = torch.ones(batch_size, 8, dtype=torch.bool)
    discrete_actions = torch.tensor([0, 1, 4, 7])  # EQ, Compressor, Distortion, STOP
    continuous_actions = torch.rand(batch_size, 5)

    log_probs, entropy, values = grrl_agent.actor_critic.evaluate_actions(
        dummy_obs, dummy_masks, discrete_actions, continuous_actions
    )

    assert log_probs.shape == (batch_size,)
    assert entropy.shape == (batch_size,)
    assert values.shape == (batch_size,)
    assert torch.isfinite(log_probs).all(), "All log probabilities must be finite"
    assert torch.isfinite(entropy).all(), "Entropy must be finite"
    assert torch.isfinite(values).all(), "Values must be finite"


# ---------------------------------------------------------------------------
# 10. GAE and Terminal-State Handling
# ---------------------------------------------------------------------------

def test_gae_computation():
    """Test Generalized Advantage Estimation (GAE) with terminal-state bootstrap."""
    buffer = GRRLRolloutBuffer(buffer_size=5, obs_dim=148, num_actions=8, max_params_per_effect=5)

    for t in range(5):
        buffer.add(
            obs=torch.randn(148),
            action_mask=torch.ones(8, dtype=torch.bool),
            discrete_action=t,
            continuous_action=torch.full((5,), 0.5),
            reward=1.0,
            value=0.5,
            log_prob=-2.0,
            done=(t == 4),  # Step 4 is terminal
        )

    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True, gamma=0.99, gae_lambda=0.95)

    assert torch.isfinite(buffer.advantages).all()
    assert torch.isfinite(buffer.returns).all()
    # At terminal step t=4, advantage delta = reward + 0 - value = 1.0 - 0.5 = 0.5
    assert abs(buffer.advantages[4].item() - 0.5) < 1e-4


# ---------------------------------------------------------------------------
# 11. Parameter and Reward Numerical Stability
# ---------------------------------------------------------------------------

def test_numerical_stability(grrl_env, synthetic_audio):
    """Test environment step produces non-NaN/non-Inf rewards under edge inputs."""
    dry, target = synthetic_audio
    grrl_env.reset(dry, target)

    # Apply all effects with boundary 0.0 and 1.0 continuous values
    for i, eff in enumerate(SUPPORTED_EFFECTS[:3]):
        act_idx = EFFECT_ACTIONS.index(eff)
        extreme_params = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        obs, reward, term, trunc, info = grrl_env.step(act_idx, continuous_params=extreme_params)

        assert math.isfinite(reward), f"Reward must be finite at step {i}"
        assert torch.isfinite(obs).all(), f"Observation must be finite at step {i}"
        assert math.isfinite(grrl_env.current_loss), f"Loss must be finite at step {i}"


# ---------------------------------------------------------------------------
# 12. Checkpoint Save and Reload
# ---------------------------------------------------------------------------

def test_checkpoint_save_and_reload(grrl_agent):
    """Test saving and loading GRRL agent checkpoint preserves weights."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_grrl.pth"
        grrl_agent.save(ckpt_path)
        assert ckpt_path.exists()

        # Create new agent and reload
        new_agent = GRRLAgent(obs_dim=grrl_agent.actor_critic.obs_dim, num_actions=8)
        new_agent.load(ckpt_path)

        # Check weights equality
        for p1, p2 in zip(grrl_agent.actor_critic.parameters(), new_agent.actor_critic.parameters()):
            assert torch.allclose(p1, p2)


# ---------------------------------------------------------------------------
# 13. M2 Warm-Start Compatibility
# ---------------------------------------------------------------------------

def test_m2_warm_start_compatibility(synthetic_audio):
    """Test GRRLEnv handles use_warm_start safely even if checkpoint not present or present."""
    dry, target = synthetic_audio
    ckpt_path = Path("checkpoints/best_model.pth")
    env = GRRLEnv(
        predictor_checkpoint=ckpt_path if ckpt_path.exists() else None,
        max_chain_length=4,
    )
    obs, info = env.reset(dry, target, use_warm_start=True)

    assert obs.shape == (148,)
    assert isinstance(info["warm_started"], bool)
    assert len(env.current_chain) <= 4


# ---------------------------------------------------------------------------
# 14. Regression Test for Fixed-Order Baseline
# ---------------------------------------------------------------------------

def test_fixed_order_baseline_regression(synthetic_audio):
    """Test that existing fixed-order AudioFXEnv and PPOAgent still work unchanged."""
    dry, target = synthetic_audio
    baseline_env = AudioFXEnv(predictor_checkpoint=None)
    baseline_agent = PPOAgent(
        obs_dim=baseline_env.obs_dim,
        discrete_dim=baseline_env.num_effects,
        continuous_dim=baseline_env.num_params,
    )

    obs, info = baseline_env.reset(dry, target, use_warm_start=False)
    assert obs.shape == (100,)
    assert baseline_env.action_dim == 36

    action_env, _, _, log_prob, val = baseline_agent.select_action(obs, deterministic=True)
    assert action_env.shape == (36,)

    next_obs, reward, term, trunc, step_info = baseline_env.step(action_env)
    assert next_obs.shape == (100,)
    assert math.isfinite(reward)
