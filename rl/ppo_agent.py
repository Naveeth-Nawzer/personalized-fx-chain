"""
rl/ppo_agent.py
===============
Actor-Critic policy architecture and PPO algorithm for DAPN audio FX chain synthesis.

Features:
- Hybrid action policy:
    1. Independent Bernoulli distribution for 7 discrete effect presence toggles.
    2. Continuous Gaussian distribution for 29 knob parameter adjustments.
- State-value function baseline V(s).
- Clipped surrogate PPO objective with entropy regularization and gradient clipping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

from rl.rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


class ActorCritic(nn.Module):
    """Hybrid discrete-continuous Actor-Critic network for AudioFXEnv.

    Parameters
    ----------
    obs_dim : int
        Observation vector dimensionality (default: 100).
    discrete_dim : int
        Number of discrete effect presence controls (default: 7).
    continuous_dim : int
        Number of continuous parameter controls (default: 29).
    hidden_dim : int
        Hidden dimension of shared feature extractor (default: 128).
    """

    def __init__(
        self,
        obs_dim: int = 100,
        discrete_dim: int = 7,
        continuous_dim: int = 29,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.discrete_dim = discrete_dim
        self.continuous_dim = continuous_dim

        # Shared representation trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # 1. Discrete effect-presence policy head (logits for 7 Bernoulli distributions)
        self.presence_head = nn.Linear(hidden_dim, discrete_dim)

        # 2. Continuous knob delta policy head (mean and trainable log-std for 29 Gaussian distributions)
        self.param_mean_head = nn.Linear(hidden_dim, continuous_dim)
        self.param_log_std = nn.Parameter(torch.full((continuous_dim,), -0.5, dtype=torch.float32))

        # 3. State-value head V(s)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute presence logits, parameter means, parameter stds, and state value."""
        features = self.shared_trunk(obs)
        presence_logits = self.presence_head(features)
        # Scale mean adjustments into [-0.5, 0.5] for smooth DAPN knob modulation
        param_mean = 0.5 * torch.tanh(self.param_mean_head(features))
        param_std = torch.exp(torch.clamp(self.param_log_std, -20.0, 2.0)).expand_as(param_mean)
        value = self.value_head(features).squeeze(-1)
        return presence_logits, param_mean, param_std, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample actions from current policy given observation(s).

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor, shape (100,) or (B, 100).
        deterministic : bool
            If True, selects mode of distributions (no sampling noise).

        Returns
        -------
        action_env : torch.Tensor
            36-dim combined action vector ready for AudioFXEnv.step().
        discrete_action : torch.Tensor
            7-dim binary vector in {0.0, 1.0}.
        continuous_action : torch.Tensor
            29-dim continuous adjustment vector.
        log_prob : torch.Tensor
            Joint log-probability log P(discrete, continuous | obs).
        value : torch.Tensor
            Estimated state value V(s).
        """
        was_1d = obs.ndim == 1
        if was_1d:
            obs = obs.unsqueeze(0)

        presence_logits, param_mean, param_std, value = self.forward(obs)

        dist_discrete = Bernoulli(logits=presence_logits)
        dist_continuous = Normal(param_mean, param_std)

        if deterministic:
            # Deterministic mode: presence thresholded at 0 logits (prob >= 0.5), continuous at mean
            discrete_action = (presence_logits > 0.0).float()
            continuous_action = param_mean
        else:
            discrete_action = dist_discrete.sample()
            continuous_action = dist_continuous.sample()

        log_prob_discrete = dist_discrete.log_prob(discrete_action).sum(dim=-1)
        log_prob_continuous = dist_continuous.log_prob(continuous_action).sum(dim=-1)
        total_log_prob = log_prob_discrete + log_prob_continuous

        # Form 36-dim action vector for AudioFXEnv:
        # First 7 are presence actions (1.0 = active, 0.0 = bypass), followed by 29 knob deltas
        action_env = torch.cat([discrete_action, continuous_action], dim=-1)

        if was_1d:
            action_env = action_env.squeeze(0)
            discrete_action = discrete_action.squeeze(0)
            continuous_action = continuous_action.squeeze(0)
            total_log_prob = total_log_prob.squeeze(0)
            value = value.squeeze(0)

        return action_env, discrete_action, continuous_action, total_log_prob, value

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        discrete_actions: torch.Tensor,
        continuous_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate joint log-probabilities, total entropy, and state values for PPO updates."""
        presence_logits, param_mean, param_std, value = self.forward(obs)

        dist_discrete = Bernoulli(logits=presence_logits)
        dist_continuous = Normal(param_mean, param_std)

        log_prob_discrete = dist_discrete.log_prob(discrete_actions).sum(dim=-1)
        log_prob_continuous = dist_continuous.log_prob(continuous_actions).sum(dim=-1)
        total_log_prob = log_prob_discrete + log_prob_continuous

        entropy_discrete = dist_discrete.entropy().sum(dim=-1)
        entropy_continuous = dist_continuous.entropy().sum(dim=-1)
        total_entropy = entropy_discrete + entropy_continuous

        return total_log_prob, total_entropy, value


class PPOAgent:
    """Proximal Policy Optimization (PPO) agent for DAPN audio chain synthesis.

    Parameters
    ----------
    obs_dim : int
        Observation space dimension (default: 100).
    discrete_dim : int
        Discrete action dimension (default: 7).
    continuous_dim : int
        Continuous action dimension (default: 29).
    lr : float
        Learning rate for Adam optimizer (default: 3e-4).
    clip_eps : float
        PPO surrogate clip threshold epsilon (default: 0.2).
    value_coef : float
        Value loss weight coefficient (default: 0.5).
    entropy_coef : float
        Entropy bonus regularization coefficient (default: 0.01).
    max_grad_norm : float
        Maximum gradient norm for gradient clipping (default: 0.5).
    device : torch.device | str
        Compute device.
    """

    def __init__(
        self,
        obs_dim: int = 100,
        discrete_dim: int = 7,
        continuous_dim: int = 29,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.actor_critic = ActorCritic(
            obs_dim=obs_dim,
            discrete_dim=discrete_dim,
            continuous_dim=continuous_dim,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
        """Select action during environment interaction.

        Returns
        -------
        action_env : torch.Tensor
            36-dim action for AudioFXEnv.
        discrete_action : torch.Tensor
            7-dim discrete presence vector.
        continuous_action : torch.Tensor
            29-dim continuous parameter adjustments.
        log_prob : float
            Scalar joint log-probability.
        value : float
            Scalar estimated state value.
        """
        self.actor_critic.eval()
        with torch.no_grad():
            action_env, disc_act, cont_act, log_prob, value = self.actor_critic.get_action(
                obs.to(self.device), deterministic=deterministic
            )
        return action_env, disc_act, cont_act, float(log_prob.item()), float(value.item())

    def train_step(
        self,
        buffer: RolloutBuffer,
        batch_size: int = 64,
        epochs: int = 4,
    ) -> dict[str, float]:
        """Execute PPO optimization over collected rollout buffer.

        Parameters
        ----------
        buffer : RolloutBuffer
            Populated rollout buffer with computed advantages and returns.
        batch_size : int
            Mini-batch size.
        epochs : int
            Number of optimization epochs over the buffer.

        Returns
        -------
        dict[str, float]
            Dictionary of training diagnostics (loss, policy_loss, value_loss, entropy, approx_kl, clip_fraction).
        """
        self.actor_critic.train()

        total_loss_accum = 0.0
        policy_loss_accum = 0.0
        value_loss_accum = 0.0
        entropy_accum = 0.0
        approx_kl_accum = 0.0
        clip_fraction_accum = 0.0
        n_batches = 0

        for _ in range(epochs):
            for batch in buffer.get_minibatches(batch_size=batch_size, normalize_advantages=True):
                obs = batch["observations"]
                disc_act = batch["discrete_actions"]
                cont_act = batch["continuous_actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Forward evaluation
                new_log_probs, entropy, values = self.actor_critic.evaluate_actions(obs, disc_act, cont_act)

                # Ratio r_t(theta) = exp(new_log_prob - old_log_prob)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(log_ratio)

                # PPO Clipped Surrogate Loss
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Function Loss (MSE)
                value_loss = 0.5 * torch.mean((values - returns).pow(2))

                # Entropy Bonus
                entropy_mean = entropy.mean()

                # Combined Objective: minimize policy_loss + c1 * value_loss - c2 * entropy
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_mean

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Diagnostics
                with torch.no_grad():
                    approx_kl = torch.mean((ratio - 1.0) - log_ratio).item()
                    clipped = (ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)
                    clip_frac = torch.mean(clipped.float()).item()

                total_loss_accum += float(loss.item())
                policy_loss_accum += float(policy_loss.item())
                value_loss_accum += float(value_loss.item())
                entropy_accum += float(entropy_mean.item())
                approx_kl_accum += approx_kl
                clip_fraction_accum += clip_frac
                n_batches += 1

        if n_batches == 0:
            return {}

        return {
            "loss": total_loss_accum / n_batches,
            "policy_loss": policy_loss_accum / n_batches,
            "value_loss": value_loss_accum / n_batches,
            "entropy": entropy_accum / n_batches,
            "approx_kl": approx_kl_accum / n_batches,
            "clip_fraction": clip_fraction_accum / n_batches,
        }

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_critic_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            str(path),
        )
        logger.info("Saved PPO checkpoint to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PPO Checkpoint not found: {path}")
        ckpt = torch.load(str(path), map_location=self.device)
        self.actor_critic.load_state_dict(ckpt["actor_critic_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Loaded PPO checkpoint from %s", path)
