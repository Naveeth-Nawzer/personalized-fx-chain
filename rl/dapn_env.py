"""
rl/dapn_env.py
==============
Goal-Conditioned Audio Effects Chain Environment for Reinforcement Learning.

Provides an environment for training an RL agent to:
1. Interactively fine-tune DSP knob parameters.
2. Optimize effect chain ordering and prune unnecessary effects (sparsity).
3. Maximize acoustic spectral matching against a goal target sound.

State Space (Observations):
---------------------------
- Feature 1: Acoustic distance embedding between Current Audio and Target Audio.
- Feature 2: Effect Presence Vector (7 binary flags for EQ, Comp, Delay, Reverb, Distortion, Chorus, Flanger).
- Feature 3: Current Continuous Parameter Values (29 continuous values in [0, 1]).

Action Space:
-------------
Continuous Action vector of dimension 36:
- Indices 0..6: Presence logits for the 7 effects (> 0.0 activates, <= 0.0 bypasses).
- Indices 7..35: Knob parameter adjustments Delta in [-0.05, +0.05] applied to current knob values.

Reward Function:
----------------
R_t = (Loss_{t-1} - Loss_t) * scale - lambda_sparse * N_active_effects + R_bonus
"""

from __future__ import annotations

import copy
import logging
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from common.losses import MultiResolutionSTFTLoss
from dapn.dapn import DAPN
from dataset.dapn_dataset import PARAM_OFFSETS, PARAM_SCHEMA, SUPPORTED_EFFECTS
from models.inverse_fx_net import InverseFXNet


class AudioFXEnv:
    """Goal-Conditioned Reinforcement Learning Environment for Audio FX Chain Synthesis."""

    def __init__(
        self,
        predictor_checkpoint: str | Path | None = "checkpoints/best_model.pth",
        device: str = "cpu",
        max_steps_per_episode: int = 15,
        sparsity_weight: float = 0.02,
        target_threshold_loss: float = 0.40,
        sample_rate: int = 44100,
        audio_duration_sec: float = 2.0,
    ) -> None:
        self.device = torch.device(device)
        self.max_steps = max_steps_per_episode
        self.sparsity_weight = sparsity_weight
        self.target_threshold_loss = target_threshold_loss
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * audio_duration_sec)

        # DSP Engine and Loss Function
        self.dapn = DAPN()
        self.loss_fn = MultiResolutionSTFTLoss()

        # Load warm-start predictor model if checkpoint provided
        self.predictor: InverseFXNet | None = None
        if predictor_checkpoint and Path(predictor_checkpoint).exists():
            ckpt = torch.load(predictor_checkpoint, map_location=self.device)
            self.predictor = InverseFXNet().to(self.device)
            self.predictor.load_state_dict(ckpt["model_state_dict"])
            self.predictor.eval()

        # Dimensions
        self.num_effects = len(SUPPORTED_EFFECTS)
        self.num_params = sum(len(p) for p in PARAM_SCHEMA.values())  # 29
        self.action_dim = self.num_effects + self.num_params  # 36
        self.obs_dim = 64 + self.num_effects + self.num_params  # Acoustic embedding (64) + Presence (7) + Params (29) = 100

        # State tracking
        self.dry_audio: torch.Tensor | None = None
        self.target_audio: torch.Tensor | None = None
        self.current_audio: torch.Tensor | None = None
        self.current_presence: torch.Tensor = torch.zeros(self.num_effects, device=self.device)
        self.current_params: torch.Tensor = torch.full((self.num_params,), 0.5, device=self.device)
        self.current_loss: float = 999.0
        self.initial_loss: float = 999.0
        self.current_step: int = 0

    def reset(
        self,
        dry_audio: torch.Tensor,
        target_audio: torch.Tensor,
        use_warm_start: bool = True,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset environment with new dry and target audio pair."""
        # Ensure correct shape: (1, 1, T)
        dry = dry_audio.to(self.device)
        target = target_audio.to(self.device)
        if dry.ndim == 1:
            dry = dry.unsqueeze(0).unsqueeze(0)
        elif dry.ndim == 2:
            dry = dry.unsqueeze(0)
        if target.ndim == 1:
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.ndim == 2:
            target = target.unsqueeze(0)

        # Crop to max duration
        self.dry_audio = dry[..., :self.max_samples]
        self.target_audio = target[..., :self.max_samples]
        self.current_step = 0

        # Initial baseline loss (dry vs target without any effects)
        with torch.no_grad():
            self.initial_loss = float(self.loss_fn(self.dry_audio, self.target_audio).item())

        # Warm-start policy initialization using InverseFXNet
        if use_warm_start and self.predictor is not None:
            with torch.no_grad():
                logits, pred_params = self.predictor(self.dry_audio, self.target_audio)
                probs = torch.sigmoid(logits)[0]
                self.current_presence = (probs >= 0.5).float()
                self.current_params = torch.clamp(pred_params[0], 0.0, 1.0)
        else:
            self.current_presence = torch.zeros(self.num_effects, device=self.device)
            self.current_params = torch.full((self.num_params,), 0.5, device=self.device)

        # Render current sound
        self._render_current_sound()
        with torch.no_grad():
            self.current_loss = float(self.loss_fn(self.current_audio, self.target_audio).item())

        obs = self._get_observation()
        info = {
            "initial_loss": self.initial_loss,
            "current_loss": self.current_loss,
            "num_active_effects": int(self.current_presence.sum().item()),
            "warm_started": use_warm_start and (self.predictor is not None),
        }
        return obs, info

    def step(
        self,
        action: torch.Tensor | list[float],
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        """Execute one RL action step."""
        if isinstance(action, (list, tuple)):
            action = torch.tensor(action, dtype=torch.float32, device=self.device)
        else:
            action = action.to(self.device)

        prev_loss = self.current_loss
        self.current_step += 1

        # Decompose action
        presence_actions = action[:self.num_effects]
        param_deltas = action[self.num_effects:]  # delta in [-0.05, +0.05]

        # Update presence: toggle on if action > 0.0, off if action <= 0.0
        self.current_presence = (presence_actions > 0.0).float()

        # Update continuous knob parameters with small adjustments: clamped in [0, 1]
        self.current_params = torch.clamp(self.current_params + param_deltas * 0.10, 0.0, 1.0)

        # Render audio through DAPN DSP
        self._render_current_sound()

        # Compute new spectral distance loss
        with torch.no_grad():
            try:
                loss_tensor = self.loss_fn(self.current_audio, self.target_audio)
                new_loss = float(loss_tensor.item())
            except Exception as e:
                logger.warning("Loss computation encountered an error: %s", e)
                new_loss = float("nan")

        if not math.isfinite(new_loss):
            new_loss = prev_loss + 1.0  # Safe penalty fallback
        self.current_loss = new_loss

        # Calculate Reward
        # 1. Spectral improvement reward: positive when loss decreases
        spectral_gain = (prev_loss - new_loss) * 10.0

        # 2. Chain sparsity penalty: encourages using fewer effects
        num_active = float(self.current_presence.sum().item())
        sparsity_penalty = self.sparsity_weight * num_active

        # 3. Target achievement bonus
        target_bonus = 5.0 if new_loss <= self.target_threshold_loss else 0.0

        reward = spectral_gain - sparsity_penalty + target_bonus
        if not math.isfinite(reward):
            reward = -1.0

        # Check termination
        terminated = bool(new_loss <= self.target_threshold_loss)
        truncated = bool(self.current_step >= self.max_steps)

        obs = self._get_observation()
        info = {
            "step": self.current_step,
            "prev_loss": prev_loss,
            "current_loss": new_loss,
            "spectral_gain": spectral_gain,
            "num_active_effects": int(num_active),
            "improvement_pct": max(0.0, (1.0 - new_loss / (self.initial_loss + 1e-8)) * 100.0),
        }

        return obs, reward, terminated, truncated, info

    def _render_current_sound(self) -> None:
        """Render audio using current presence flags and parameter settings through DAPN."""
        chain_steps = []
        for e_idx, effect_name in enumerate(SUPPORTED_EFFECTS):
            if self.current_presence[e_idx] > 0.5:
                start_idx, end_idx = PARAM_OFFSETS[effect_name]
                param_names = PARAM_SCHEMA[effect_name]
                eff_params = self.current_params[start_idx:end_idx]

                params_dict = {
                    p_name: float(eff_params[i].item())
                    for i, p_name in enumerate(param_names)
                }
                chain_steps.append({"effect": effect_name, "parameters": params_dict})

        chain_dict = {"chain": chain_steps}
        with torch.no_grad():
            if chain_steps:
                try:
                    rendered = self.dapn.process(self.dry_audio, chain_dict)
                    if not torch.isfinite(rendered).all():
                        logger.warning("Rendered audio contains non-finite values; using previous audio.")
                        if self.current_audio is None or not torch.isfinite(self.current_audio).all():
                            self.current_audio = self.dry_audio.clone()
                    else:
                        self.current_audio = rendered
                except Exception as e:
                    logger.warning("Audio processing failed: %s; falling back to previous state.", e)
                    if self.current_audio is None:
                        self.current_audio = self.dry_audio.clone()
            else:
                self.current_audio = self.dry_audio.clone()

    def _get_observation(self) -> torch.Tensor:
        """Construct compact state representation for RL agent."""
        with torch.no_grad():
            # Acoustic difference representation: downsampled spectral difference
            n_fft = 512
            win = torch.hann_window(n_fft, device=self.device)
            c_stft = torch.stft(self.current_audio.squeeze(0), n_fft=n_fft, hop_length=256, window=win, return_complex=True).abs()
            t_stft = torch.stft(self.target_audio.squeeze(0), n_fft=n_fft, hop_length=256, window=win, return_complex=True).abs()

            # Mean frequency profile difference: 64 features
            diff = (t_stft - c_stft).mean(dim=-1).squeeze(0)
            if diff.shape[-1] >= 64:
                acoustic_feats = F.adaptive_avg_pool1d(diff.unsqueeze(0).unsqueeze(0), 64).squeeze()
            else:
                acoustic_feats = F.pad(diff, (0, 64 - diff.shape[-1]))

        # Concatenate acoustic embedding + current presence (7) + current params (29) = 100 dims
        obs = torch.cat([acoustic_feats, self.current_presence, self.current_params], dim=0)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        return obs

    def get_current_chain_summary(self) -> dict[str, Any]:
        """Returns clean dictionary summary of the currently configured chain."""
        active_chain = []
        for e_idx, effect_name in enumerate(SUPPORTED_EFFECTS):
            if self.current_presence[e_idx] > 0.5:
                start_idx, end_idx = PARAM_OFFSETS[effect_name]
                param_names = PARAM_SCHEMA[effect_name]
                params_dict = {
                    p_name: round(float(self.current_params[start_idx + i].item()), 4)
                    for i, p_name in enumerate(param_names)
                }
                active_chain.append({"effect": effect_name, "parameters": params_dict})
        return {
            "chain": active_chain,
            "loss": round(self.current_loss, 5),
            "num_effects": len(active_chain),
        }
