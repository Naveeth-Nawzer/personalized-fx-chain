"""
rl/grrl_agent.py
================
Actor-Critic policy architecture and PPO algorithm for the Dynamic
Graph-Generating Reinforcement Learning Agent (GRRL-Agent).

Features:
- Hybrid action space:
    1. Masked Categorical distribution over 8 discrete actions:
       {EQ, Compressor, Delay, Reverb, Distortion, Chorus, Flanger, STOP}.
    2. Per-effect Gaussian continuous parameter heads generating normalized
       knob values strictly bounded in [0, 1].
- Exact joint log-probability P(effect, params | state).
- State-value function baseline V(s).
- Clipped surrogate PPO objective with entropy regularization and gradient clipping.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

from dataset.dapn_dataset import PARAM_SCHEMA, SUPPORTED_EFFECTS
from rl.grrl_buffer import GRRLRolloutBuffer
from rl.grrl_env import EFFECT_ACTIONS, MAX_PARAMS_PER_EFFECT, NUM_EFFECT_ACTIONS, STOP_ACTION_IDX

logger = logging.getLogger(__name__)


class GRRLActorCritic(nn.Module):
    """Hybrid Masked Categorical + Per-Effect Continuous Actor-Critic network for GRRL-Agent.

    Parameters
    ----------
    obs_dim : int
        Observation vector dimensionality (default: 148).
    num_actions : int
        Number of discrete actions (default: 8).
    hidden_dim : int
        Hidden dimension of shared feature trunk (default: 128).
    """

    def __init__(
        self,
        obs_dim: int = 148,
        num_actions: int = 8,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.hidden_dim = hidden_dim

        # Shared representation trunk
        self.shared_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # 1. Discrete Policy Head: Logits for 8 actions (7 effects + STOP)
        self.discrete_head = nn.Linear(hidden_dim, num_actions)

        # 2. Continuous Parameter Policy Heads: One dedicated head per effect
        self.param_heads = nn.ModuleDict({
            eff: nn.Linear(hidden_dim, len(PARAM_SCHEMA[eff]))
            for eff in SUPPORTED_EFFECTS
        })
        self.param_log_stds = nn.ParameterDict({
            eff: nn.Parameter(torch.full((len(PARAM_SCHEMA[eff]),), -0.5, dtype=torch.float32))
            for eff in SUPPORTED_EFFECTS
        })

        # 3. State-Value Head V(s)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract latent features from shared trunk."""
        return self.shared_trunk(obs)

    def get_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, float, float]:
        """Sample sequential hybrid action for environment interaction.

        Parameters
        ----------
        obs : torch.Tensor
            Observation tensor, shape (obs_dim,).
        action_mask : torch.Tensor, optional
            Boolean validity mask, shape (num_actions,). True = valid.
        deterministic : bool
            If True, uses argmax for discrete action and mean for continuous parameters.

        Returns
        -------
        discrete_action : int
            Index of chosen action in [0, 7].
        continuous_action : torch.Tensor
            Parameter vector for chosen effect, shape (max_params_per_effect=5,).
        log_prob : float
            Scalar joint log-probability log P(effect, params | obs).
        value : float
            Estimated state value V(s).
        """
        was_1d = obs.ndim == 1
        if was_1d:
            obs = obs.unsqueeze(0)
            if action_mask is not None and action_mask.ndim == 1:
                action_mask = action_mask.unsqueeze(0)

        features = self.forward_features(obs)
        discrete_logits = self.discrete_head(features)
        value = self.value_head(features).squeeze(-1)

        # Apply action mask if provided
        if action_mask is not None:
            # Mask invalid actions with a large negative number
            masked_logits = discrete_logits.masked_fill(~action_mask, -1e9)
        else:
            masked_logits = discrete_logits

        dist_discrete = Categorical(logits=masked_logits)

        if deterministic:
            discrete_act = torch.argmax(masked_logits, dim=-1)
        else:
            discrete_act = dist_discrete.sample()

        log_prob_discrete = dist_discrete.log_prob(discrete_act)

        # Continuous parameters for selected discrete action
        disc_item = int(discrete_act[0].item())
        cont_act = torch.zeros(MAX_PARAMS_PER_EFFECT, dtype=torch.float32, device=obs.device)
        log_prob_cont = torch.tensor(0.0, device=obs.device)

        if disc_item != STOP_ACTION_IDX:
            eff_name = EFFECT_ACTIONS[disc_item]
            head = self.param_heads[eff_name]
            log_std = self.param_log_stds[eff_name]

            # Param mean bounded in [0.0, 1.0] via sigmoid
            param_mean = torch.sigmoid(head(features[0]))
            param_std = torch.exp(torch.clamp(log_std, -20.0, 2.0))
            dist_cont = Normal(param_mean, param_std)

            if deterministic:
                sample_p = param_mean
            else:
                sample_p = dist_cont.sample()

            sample_p = torch.clamp(sample_p, 0.0, 1.0)
            log_prob_cont = dist_cont.log_prob(sample_p).sum()
            cont_act[:len(sample_p)] = sample_p

        total_log_prob = log_prob_discrete[0] + log_prob_cont

        return disc_item, cont_act, float(total_log_prob.item()), float(value[0].item())

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action_masks: torch.Tensor,
        discrete_actions: torch.Tensor,
        continuous_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate joint log-probabilities, total entropy, and state values for PPO updates.

        Parameters
        ----------
        obs : torch.Tensor
            Batch observations, shape (B, obs_dim).
        action_masks : torch.Tensor
            Batch action masks, shape (B, num_actions).
        discrete_actions : torch.Tensor
            Batch discrete actions, shape (B,).
        continuous_actions : torch.Tensor
            Batch continuous actions, shape (B, max_params_per_effect).

        Returns
        -------
        total_log_probs : torch.Tensor
            Joint log probabilities, shape (B,).
        total_entropy : torch.Tensor
            Sum of discrete and continuous entropies, shape (B,).
        values : torch.Tensor
            Estimated state values V(s), shape (B,).
        """
        B = obs.shape[0]
        features = self.forward_features(obs)
        discrete_logits = self.discrete_head(features)
        values = self.value_head(features).squeeze(-1)

        # Apply action masks
        masked_logits = discrete_logits.masked_fill(~action_masks, -1e9)
        dist_discrete = Categorical(logits=masked_logits)

        log_prob_discrete = dist_discrete.log_prob(discrete_actions)  # (B,)
        entropy_discrete = dist_discrete.entropy()                   # (B,)

        log_prob_cont = torch.zeros(B, dtype=torch.float32, device=obs.device)
        entropy_cont = torch.zeros(B, dtype=torch.float32, device=obs.device)

        # Vectorized continuous head evaluation per effect type
        for eff_idx, eff_name in enumerate(SUPPORTED_EFFECTS):
            mask_eff = (discrete_actions == eff_idx)
            if mask_eff.any():
                sub_feats = features[mask_eff]
                k_params = len(PARAM_SCHEMA[eff_name])

                param_mean = torch.sigmoid(self.param_heads[eff_name](sub_feats))
                param_std = torch.exp(torch.clamp(self.param_log_stds[eff_name], -20.0, 2.0)).expand_as(param_mean)
                dist_sub = Normal(param_mean, param_std)

                sub_actions = continuous_actions[mask_eff, :k_params]
                # Log prob of continuous parameters
                lp_sub = dist_sub.log_prob(sub_actions).sum(dim=-1)
                ent_sub = dist_sub.entropy().sum(dim=-1)

                log_prob_cont[mask_eff] = lp_sub
                entropy_cont[mask_eff] = ent_sub

        total_log_probs = log_prob_discrete + log_prob_cont
        total_entropy = entropy_discrete + entropy_cont

        return total_log_probs, total_entropy, values


class GRRLAgent:
    """Proximal Policy Optimization (PPO) agent for dynamic sequential FX-chain synthesis.

    Parameters
    ----------
    obs_dim : int
        Observation space dimension (default: 148).
    num_actions : int
        Discrete action space dimension (default: 8).
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
        obs_dim: int = 148,
        num_actions: int = 8,
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

        self.actor_critic = GRRLActorCritic(
            obs_dim=obs_dim,
            num_actions=num_actions,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)

    def select_action(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[int, torch.Tensor, float, float]:
        """Select action during environment interaction.

        Returns
        -------
        discrete_action : int
            Index of chosen effect or STOP.
        continuous_action : torch.Tensor
            Parameter vector for chosen effect.
        log_prob : float
            Scalar joint log-probability.
        value : float
            Scalar estimated state value.
        """
        self.actor_critic.eval()
        with torch.no_grad():
            mask = action_mask.to(self.device) if action_mask is not None else None
            disc_act, cont_act, log_prob, value = self.actor_critic.get_action(
                obs.to(self.device), action_mask=mask, deterministic=deterministic
            )
        return disc_act, cont_act, log_prob, value

    def train_step(
        self,
        buffer: GRRLRolloutBuffer,
        batch_size: int = 64,
        epochs: int = 4,
    ) -> dict[str, float]:
        """Execute PPO optimization over collected rollout buffer.

        Parameters
        ----------
        buffer : GRRLRolloutBuffer
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
                action_masks = batch["action_masks"]
                disc_act = batch["discrete_actions"]
                cont_act = batch["continuous_actions"]
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]
                returns = batch["returns"]

                # Forward evaluation
                new_log_probs, entropy, values = self.actor_critic.evaluate_actions(
                    obs, action_masks, disc_act, cont_act
                )

                # Ratio r_t(theta) = exp(new_log_prob - old_log_prob)
                log_ratio = new_log_probs - old_log_probs
                ratio = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))

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
        logger.info("Saved GRRL checkpoint to %s", path)

    def load(self, path: str | Path) -> None:
        """Load model checkpoint."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"GRRL Checkpoint not found: {path}")
        ckpt = torch.load(str(path), map_location=self.device)
        self.actor_critic.load_state_dict(ckpt["actor_critic_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        logger.info("Loaded GRRL checkpoint from %s", path)
