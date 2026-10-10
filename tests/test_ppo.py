"""
tests/test_ppo.py
=================
Unit and integration tests for M3 PPO Reinforcement Learning Agent, RolloutBuffer,
Generalized Advantage Estimation (GAE), and AudioFXEnv integration.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
import pytest
import torch

from rl.dapn_env import AudioFXEnv
from rl.ppo_agent import ActorCritic, PPOAgent
from rl.rollout_buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dummy_obs_1d():
    torch.manual_seed(42)
    return torch.randn(100)


@pytest.fixture
def dummy_obs_batch():
    torch.manual_seed(42)
    return torch.randn(4, 100)


@pytest.fixture
def actor_critic():
    torch.manual_seed(42)
    return ActorCritic(obs_dim=100, discrete_dim=7, continuous_dim=29, hidden_dim=64)


# ---------------------------------------------------------------------------
# 1. Actor-Critic Architecture & Shapes
# ---------------------------------------------------------------------------

def test_actor_critic_forward_shapes(actor_critic, dummy_obs_batch):
    logits, mean, std, val = actor_critic(dummy_obs_batch)

    assert logits.shape == (4, 7), f"Expected presence logits (4, 7), got {logits.shape}"
    assert mean.shape == (4, 29), f"Expected parameter mean (4, 29), got {mean.shape}"
    assert std.shape == (4, 29), f"Expected parameter std (4, 29), got {std.shape}"
    assert val.shape == (4,), f"Expected state values (4,), got {val.shape}"
    assert torch.all(std > 0.0), "Standard deviation must be strictly positive"


def test_actor_critic_get_action_1d(actor_critic, dummy_obs_1d):
    action_env, disc_act, cont_act, log_prob, val = actor_critic.get_action(dummy_obs_1d, deterministic=False)

    assert action_env.shape == (36,), f"Expected action_env (36,), got {action_env.shape}"
    assert disc_act.shape == (7,), f"Expected discrete_act (7,), got {disc_act.shape}"
    assert cont_act.shape == (29,), f"Expected continuous_act (29,), got {cont_act.shape}"
    assert log_prob.ndim == 0, f"Expected scalar log_prob, got {log_prob.ndim}"
    assert val.ndim == 0, f"Expected scalar val, got {val.ndim}"

    # Verify discrete action binary range
    assert torch.all((disc_act == 0.0) | (disc_act == 1.0))
    # Verify action_env assembly
    assert torch.allclose(action_env[:7], disc_act)
    assert torch.allclose(action_env[7:], cont_act)


def test_actor_critic_get_action_deterministic(actor_critic, dummy_obs_1d):
    action1, d1, c1, _, _ = actor_critic.get_action(dummy_obs_1d, deterministic=True)
    action2, d2, c2, _, _ = actor_critic.get_action(dummy_obs_1d, deterministic=True)

    assert torch.allclose(action1, action2), "Deterministic mode must produce identical actions"
    assert torch.allclose(d1, d2)
    assert torch.allclose(c1, c2)


def test_actor_critic_evaluate_actions(actor_critic, dummy_obs_batch):
    _, disc_act, cont_act, _, _ = actor_critic.get_action(dummy_obs_batch, deterministic=False)
    log_probs, entropy, values = actor_critic.evaluate_actions(dummy_obs_batch, disc_act, cont_act)

    assert log_probs.shape == (4,)
    assert entropy.shape == (4,)
    assert values.shape == (4,)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(entropy).all()
    assert torch.isfinite(values).all()


# ---------------------------------------------------------------------------
# 2. Generalized Advantage Estimation (GAE)
# ---------------------------------------------------------------------------

def test_gae_analytical_calculation():
    """Verify GAE and returns match manual mathematical calculation."""
    buffer = RolloutBuffer(buffer_size=3, obs_dim=10, discrete_dim=2, continuous_dim=2)

    # Synthetic transitions
    buffer.add(torch.zeros(10), torch.zeros(2), torch.zeros(2), reward=1.0, value=0.5, log_prob=-1.0, done=False)
    buffer.add(torch.zeros(10), torch.zeros(2), torch.zeros(2), reward=2.0, value=1.0, log_prob=-1.0, done=True) # terminal!
    buffer.add(torch.zeros(10), torch.zeros(2), torch.zeros(2), reward=3.0, value=1.5, log_prob=-1.0, done=False)

    gamma = 0.90
    gae_lambda = 0.95
    last_val = 2.0
    last_done = False

    buffer.compute_returns_and_advantages(last_value=last_val, last_done=last_done, gamma=gamma, gae_lambda=gae_lambda)

    # Step 1 is terminal:
    # delta_1 = reward_1 + gamma * 0 - value_1 = 2.0 - 1.0 = 1.0
    # adv_1 = delta_1 = 1.0
    assert math.isclose(buffer.advantages[1].item(), 1.0, rel_tol=1e-5)
    assert math.isclose(buffer.returns[1].item(), 2.0, rel_tol=1e-5)

    # Step 0:
    # delta_0 = reward_0 + gamma * value_1 - value_0 = 1.0 + 0.9 * 1.0 - 0.5 = 1.4
    # adv_0 = delta_0 + gamma * gae_lambda * adv_1 = 1.4 + 0.9 * 0.95 * 1.0 = 2.255
    assert math.isclose(buffer.advantages[0].item(), 2.255, rel_tol=1e-5)
    assert math.isclose(buffer.returns[0].item(), 2.755, rel_tol=1e-5)


# ---------------------------------------------------------------------------
# 3. Rollout Buffer & Mini-Batch Iteration
# ---------------------------------------------------------------------------

def test_rollout_buffer_minibatches():
    buffer = RolloutBuffer(buffer_size=10, obs_dim=20, discrete_dim=7, continuous_dim=29)

    for i in range(10):
        buffer.add(
            obs=torch.full((20,), float(i)),
            discrete_action=torch.zeros(7),
            continuous_action=torch.zeros(29),
            reward=float(i),
            value=float(i * 0.5),
            log_prob=-0.5,
            done=(i == 9),
        )

    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True)

    batches = list(buffer.get_minibatches(batch_size=4, normalize_advantages=True))
    assert len(batches) == 3  # 4 + 4 + 2 = 10
    total_samples = sum(b["observations"].shape[0] for b in batches)
    assert total_samples == 10

    # Verify advantage normalization
    all_advs = torch.cat([b["advantages"] for b in batches])
    assert abs(all_advs.mean().item()) < 1e-4
    assert abs(all_advs.std().item() - 1.0) < 1e-2


# ---------------------------------------------------------------------------
# 4. PPO Agent Optimization & Checkpointing
# ---------------------------------------------------------------------------

def test_ppo_train_step_updates_weights():
    agent = PPOAgent(obs_dim=50, discrete_dim=7, continuous_dim=29, lr=1e-3)
    buffer = RolloutBuffer(buffer_size=16, obs_dim=50, discrete_dim=7, continuous_dim=29)

    for i in range(16):
        obs = torch.randn(50)
        action_env, disc_act, cont_act, log_prob, val = agent.select_action(obs)
        buffer.add(obs, disc_act, cont_act, reward=1.0, value=val, log_prob=log_prob, done=(i % 4 == 3))

    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True)

    # Save initial parameters
    orig_params = [p.clone() for p in agent.actor_critic.parameters()]

    metrics = agent.train_step(buffer, batch_size=8, epochs=2)

    assert "loss" in metrics
    assert "policy_loss" in metrics
    assert "value_loss" in metrics
    assert "entropy" in metrics
    assert math.isfinite(metrics["loss"])

    # Verify parameters actually updated
    changed = False
    for p_orig, p_new in zip(orig_params, agent.actor_critic.parameters()):
        if not torch.allclose(p_orig, p_new, atol=1e-5):
            changed = True
            break
    assert changed, "PPO training step must update network parameters"


def test_ppo_checkpoint_save_and_load():
    agent1 = PPOAgent(obs_dim=30, discrete_dim=7, continuous_dim=29)
    agent2 = PPOAgent(obs_dim=30, discrete_dim=7, continuous_dim=29)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test_ppo.pth"
        agent1.save(ckpt_path)
        assert ckpt_path.exists()

        agent2.load(ckpt_path)

        obs = torch.randn(30)
        _, d1, c1, _, _ = agent1.select_action(obs, deterministic=True)
        _, d2, c2, _, _ = agent2.select_action(obs, deterministic=True)

        assert torch.allclose(d1, d2)
        assert torch.allclose(c1, c2)


# ---------------------------------------------------------------------------
# 5. Full AudioFXEnv Interaction Smoke Test
# ---------------------------------------------------------------------------

def test_ppo_audiofx_env_end_to_end_smoke():
    """Verify that PPO agent can run an episode and perform a PPO update with AudioFXEnv."""
    env = AudioFXEnv(predictor_checkpoint=None, max_steps_per_episode=4)
    agent = PPOAgent(obs_dim=env.obs_dim, discrete_dim=env.num_effects, continuous_dim=env.num_params)
    buffer = RolloutBuffer(buffer_size=4, obs_dim=env.obs_dim, discrete_dim=env.num_effects, continuous_dim=env.num_params)

    sr = 44100
    t = torch.linspace(0, 0.5, int(sr * 0.5))
    dry = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
    target = 0.5 * torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

    obs, _ = env.reset(dry, target, use_warm_start=False)

    for _ in range(4):
        action_env, disc_act, cont_act, log_prob, val = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action_env)
        done = terminated or truncated
        buffer.add(obs, disc_act, cont_act, reward, val, log_prob, done)
        obs = next_obs
        if done:
            break

    buffer.compute_returns_and_advantages(last_value=0.0, last_done=True)
    metrics = agent.train_step(buffer, batch_size=4, epochs=1)

    assert "loss" in metrics
    assert math.isfinite(metrics["loss"])
