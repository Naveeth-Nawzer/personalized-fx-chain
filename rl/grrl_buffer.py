"""
rl/grrl_buffer.py
=================
Rollout buffer and Generalized Advantage Estimation (GAE) for GRRL-Agent.

Supports the sequential hybrid action space:
- Observations: shape (obs_dim,)
- Action masks: shape (num_actions,)
- Discrete effect actions: scalar int in [0, num_actions - 1]
- Continuous parameters: shape (max_params_per_effect,)
- Scalar rewards, state values, log probabilities, and terminal indicators.
"""

from __future__ import annotations

from typing import Generator
import torch


class GRRLRolloutBuffer:
    """Storage buffer for on-policy rollouts of the dynamic GRRL-Agent.

    Parameters
    ----------
    buffer_size : int
        Maximum number of transition steps to store in the buffer.
    obs_dim : int
        Observation vector dimensionality (default: 148).
    num_actions : int
        Number of discrete actions (default: 8).
    max_params_per_effect : int
        Maximum number of continuous parameters for any single effect (default: 5).
    device : torch.device | str
        Target device for stored tensors.
    """

    def __init__(
        self,
        buffer_size: int,
        obs_dim: int = 148,
        num_actions: int = 8,
        max_params_per_effect: int = 5,
        device: torch.device | str = "cpu",
    ) -> None:
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self.max_params_per_effect = max_params_per_effect
        self.device = torch.device(device)

        self.reset()

    def reset(self) -> None:
        """Clear buffer and reset insertion pointer."""
        self.observations = torch.zeros((self.buffer_size, self.obs_dim), dtype=torch.float32, device=self.device)
        self.action_masks = torch.ones((self.buffer_size, self.num_actions), dtype=torch.bool, device=self.device)
        self.discrete_actions = torch.zeros(self.buffer_size, dtype=torch.long, device=self.device)
        self.continuous_actions = torch.zeros(
            (self.buffer_size, self.max_params_per_effect), dtype=torch.float32, device=self.device
        )
        self.rewards = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.values = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.log_probs = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)

        self.advantages = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(self.buffer_size, dtype=torch.float32, device=self.device)

        self.ptr = 0
        self.is_full = False

    def add(
        self,
        obs: torch.Tensor,
        action_mask: torch.Tensor,
        discrete_action: int | torch.Tensor,
        continuous_action: torch.Tensor,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Insert a single transition into the rollout buffer."""
        if self.ptr >= self.buffer_size:
            raise IndexError("GRRLRolloutBuffer overflow: attempt to add transition beyond buffer_size.")

        self.observations[self.ptr] = obs.to(self.device)
        self.action_masks[self.ptr] = action_mask.to(self.device)

        if isinstance(discrete_action, torch.Tensor):
            self.discrete_actions[self.ptr] = int(discrete_action.item())
        else:
            self.discrete_actions[self.ptr] = int(discrete_action)

        # Pad or slice continuous action to max_params_per_effect (5)
        cont_t = continuous_action.to(self.device)
        if cont_t.numel() < self.max_params_per_effect:
            padded = torch.zeros(self.max_params_per_effect, dtype=torch.float32, device=self.device)
            padded[:cont_t.numel()] = cont_t
            self.continuous_actions[self.ptr] = padded
        else:
            self.continuous_actions[self.ptr] = cont_t[:self.max_params_per_effect]

        self.rewards[self.ptr] = float(reward)
        self.values[self.ptr] = float(value)
        self.log_probs[self.ptr] = float(log_prob)
        self.dones[self.ptr] = 1.0 if done else 0.0

        self.ptr += 1
        if self.ptr == self.buffer_size:
            self.is_full = True

    def compute_returns_and_advantages(
        self,
        last_value: float,
        last_done: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalized Advantage Estimation (GAE) and discounted return targets.

        Parameters
        ----------
        last_value : float
            Value prediction for state s_{T} (bootstrap value).
        last_done : bool
            Whether the final state was terminal.
        gamma : float
            Discount factor (default: 0.99).
        gae_lambda : float
            GAE exponential weighting parameter (default: 0.95).
        """
        last_val = float(last_value)
        last_d = 1.0 if last_done else 0.0
        gae = 0.0

        for t in reversed(range(self.ptr)):
            if t == self.ptr - 1:
                next_val = last_val
                next_non_terminal = 1.0 - last_d
            else:
                next_val = self.values[t + 1].item()
                next_non_terminal = 1.0 - self.dones[t].item()

            delta = self.rewards[t].item() + gamma * next_val * next_non_terminal - self.values[t].item()
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t].item()

    def get_minibatches(
        self,
        batch_size: int = 64,
        normalize_advantages: bool = True,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield randomized mini-batches for PPO updates."""
        if self.ptr == 0:
            return

        indices = torch.randperm(self.ptr, device=self.device)

        # Normalize advantages across current valid buffer
        valid_advantages = self.advantages[:self.ptr]
        if normalize_advantages and valid_advantages.numel() > 1:
            mean = valid_advantages.mean()
            std = valid_advantages.std() + 1e-8
            normalized_advantages = (valid_advantages - mean) / std
        else:
            normalized_advantages = valid_advantages

        for start_idx in range(0, self.ptr, batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            yield {
                "observations": self.observations[batch_indices],
                "action_masks": self.action_masks[batch_indices],
                "discrete_actions": self.discrete_actions[batch_indices],
                "continuous_actions": self.continuous_actions[batch_indices],
                "old_log_probs": self.log_probs[batch_indices],
                "advantages": normalized_advantages[batch_indices],
                "returns": self.returns[batch_indices],
            }
