"""
rl/grrl_env.py
==============
Dynamic Graph-Generating Reinforcement Learning Environment (GRRL-Env)
for Sequential Audio Effects Chain Synthesis.

In contrast to the fixed-slot AudioFXEnv, this environment implements:
1. Sequential action execution: At each step, the agent chooses an effect
   from {EQ, Compressor, Delay, Reverb, Distortion, Chorus, Flanger, STOP}
   and continuous parameters for that specific effect.
2. Dynamic chain ordering: Effects are appended to `current_chain` in the
   exact order selected by the agent (e.g. Distortion -> EQ -> Reverb).
3. Explicit STOP termination: Chain synthesis ends immediately when the agent
   selects the STOP action, or when `max_chain_length` is reached.
4. Action masking: Invalid actions (e.g., repeated effects when disallowed)
   are masked before policy sampling.
5. Sequential state representation: Tracks acoustic spectral distance,
   ordered chain history, and step progress.
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

from common.config_loader import load_system_config
from common.losses import MultiResolutionSTFTLoss
from dapn.dapn import DAPN
from dataset.dapn_dataset import PARAM_OFFSETS, PARAM_SCHEMA, SUPPORTED_EFFECTS
from models.inverse_fx_net import InverseFXNet

logger = logging.getLogger(__name__)

# Action definition: 7 effects + STOP
EFFECT_ACTIONS: list[str] = list(SUPPORTED_EFFECTS) + ["STOP"]
NUM_EFFECT_ACTIONS: int = len(EFFECT_ACTIONS)  # 8
STOP_ACTION_IDX: int = NUM_EFFECT_ACTIONS - 1  # 7

MAX_PARAMS_PER_EFFECT: int = max(len(params) for params in PARAM_SCHEMA.values())  # 5
MAX_POSSIBLE_CHAIN_LENGTH: int = 8


class GRRLEnv:
    """Dynamic Graph-Generating Reinforcement Learning Environment for Audio FX Chains.

    Parameters
    ----------
    predictor_checkpoint : str | Path | None
        Path to InverseFXNet checkpoint for warm-start initialization.
    device : str
        Compute device ('cpu', 'cuda', 'mps').
    max_chain_length : int | None
        Maximum number of effects in chain (defaults to system_config.yaml: 8).
    min_chain_length : int | None
        Minimum number of effects before STOP is permitted (defaults to system_config.yaml: 1).
    allow_repeated_effects : bool | None
        Whether duplicate effect types are permitted (defaults to system_config.yaml: False).
    step_penalty : float
        Regularization penalty subtracted at each step to encourage concise chains.
    target_threshold_loss : float
        Target loss threshold for early completion bonus.
    sample_rate : int
        Audio sample rate in Hz (44100).
    audio_duration_sec : float
        Maximum audio clip length in seconds.
    pta_taste_dim : int
        Dimensionality of PTA taste conditioning vector (default: 16).
    """

    def __init__(
        self,
        predictor_checkpoint: str | Path | None = "checkpoints/best_model.pth",
        device: str = "cpu",
        max_chain_length: int | None = None,
        min_chain_length: int | None = None,
        allow_repeated_effects: bool | None = None,
        step_penalty: float = 0.05,
        target_threshold_loss: float = 0.40,
        sample_rate: int = 44100,
        audio_duration_sec: float = 2.0,
        pta_taste_dim: int = 16,
    ) -> None:
        self.device = torch.device(device)
        self.step_penalty = step_penalty
        self.target_threshold_loss = target_threshold_loss
        self.sample_rate = sample_rate
        self.max_samples = int(sample_rate * audio_duration_sec)
        self.pta_taste_dim = pta_taste_dim

        # Load system config rules
        sys_cfg = load_system_config().get("chain", {})
        self.max_chain_length = max_chain_length if max_chain_length is not None else int(sys_cfg.get("max_chain_length", 8))
        self.min_chain_length = min_chain_length if min_chain_length is not None else int(sys_cfg.get("min_chain_length", 1))
        self.allow_repeated_effects = (
            allow_repeated_effects if allow_repeated_effects is not None
            else bool(sys_cfg.get("allow_repeated_effects", False))
        )

        # DSP Engine and Loss
        self.dapn = DAPN()
        self.loss_fn = MultiResolutionSTFTLoss()

        # Warm-start predictor (M2)
        self.predictor: InverseFXNet | None = None
        if predictor_checkpoint and Path(predictor_checkpoint).exists():
            ckpt = torch.load(predictor_checkpoint, map_location=self.device)
            self.predictor = InverseFXNet().to(self.device)
            self.predictor.load_state_dict(ckpt["model_state_dict"])
            self.predictor.eval()

        # Dimensions:
        # 1. Acoustic difference embedding: 64
        # 2. Chain history: max_chain_length (8) slots * (len(SUPPORTED_EFFECTS) + 1 (EMPTY)) = 8 * 8 = 64
        # 3. Step & Loss Scalar features: 4 (step/max_steps, len(chain)/max_steps, current_loss/10, improvement_pct/100)
        # 4. PTA Taste conditioning vector: 16 (zeros by default, compatible with PTA)
        self.acoustic_dim = 64
        self.history_slots = MAX_POSSIBLE_CHAIN_LENGTH  # 8
        self.history_dim = self.history_slots * (len(SUPPORTED_EFFECTS) + 1)  # 64
        self.metrics_dim = 4
        self.obs_dim = self.acoustic_dim + self.history_dim + self.metrics_dim + self.pta_taste_dim  # 148

        self.num_actions = NUM_EFFECT_ACTIONS  # 8 (7 effects + STOP)
        self.max_params_per_effect = MAX_PARAMS_PER_EFFECT  # 5

        # Episode State Tracking
        self.dry_audio: torch.Tensor | None = None
        self.target_audio: torch.Tensor | None = None
        self.current_audio: torch.Tensor | None = None
        self.current_chain: list[dict[str, Any]] = []
        self.current_loss: float = 999.0
        self.initial_loss: float = 999.0
        self.current_step: int = 0
        self.pta_taste_vector: torch.Tensor = torch.zeros(self.pta_taste_dim, device=self.device)

    def reset(
        self,
        dry_audio: torch.Tensor,
        target_audio: torch.Tensor,
        use_warm_start: bool = False,
        pta_taste: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset the environment for a new audio target synthesis episode.

        Parameters
        ----------
        dry_audio : torch.Tensor
            Input unprocessed dry waveform, shape (C, T) or (1, C, T).
        target_audio : torch.Tensor
            Target reference audio to match.
        use_warm_start : bool
            If True and M2 predictor is loaded, populates initial chain candidate
            based on M2 predicted high-confidence effects.
        pta_taste : torch.Tensor, optional
            Optional D-dimensional taste conditioning vector from PTA.

        Returns
        -------
        obs : torch.Tensor
            State observation tensor, shape (obs_dim,).
        info : dict[str, Any]
            Diagnostic reset info dictionary.
        """
        # Format audio shapes to (1, 1, T)
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

        self.dry_audio = dry[..., :self.max_samples]
        self.target_audio = target[..., :self.max_samples]
        self.current_step = 0
        self.current_chain = []

        if pta_taste is not None:
            self.pta_taste_vector = pta_taste.to(self.device)
        else:
            self.pta_taste_vector = torch.zeros(self.pta_taste_dim, device=self.device)

        # Initial baseline loss (unprocessed dry audio vs target)
        with torch.no_grad():
            self.initial_loss = float(self.loss_fn(self.dry_audio, self.target_audio).item())
            self.current_loss = self.initial_loss
            self.current_audio = self.dry_audio.clone()

        # Optional M2 warm-start initialization
        warm_started = False
        if use_warm_start and self.predictor is not None:
            with torch.no_grad():
                logits, pred_params = self.predictor(self.dry_audio, self.target_audio)
                probs = torch.sigmoid(logits)[0]
                params = torch.clamp(pred_params[0], 0.0, 1.0)

                # Collect detected effects sorted by prediction confidence
                detected = []
                for e_idx, eff_name in enumerate(SUPPORTED_EFFECTS):
                    conf = float(probs[e_idx].item())
                    if conf >= 0.5:
                        detected.append((eff_name, conf))

                # Sort by confidence descending to form initial candidate sequence
                detected.sort(key=lambda x: x[1], reverse=True)

                if detected:
                    for eff_name, _ in detected[:self.max_chain_length]:
                        start_i, end_i = PARAM_OFFSETS[eff_name]
                        p_names = PARAM_SCHEMA[eff_name]
                        eff_p = params[start_i:end_i]
                        p_dict = {
                            p_name: float(eff_p[i].item())
                            for i, p_name in enumerate(p_names)
                        }
                        self.current_chain.append({"effect": eff_name, "parameters": p_dict})

                    self._render_current_sound()
                    self.current_loss = float(self.loss_fn(self.current_audio, self.target_audio).item())
                    warm_started = True

        obs = self._get_observation()
        info = {
            "initial_loss": self.initial_loss,
            "current_loss": self.current_loss,
            "chain_length": len(self.current_chain),
            "warm_started": warm_started,
            "action_mask": self.get_action_mask(),
        }
        return obs, info

    def get_action_mask(self) -> torch.Tensor:
        """Returns boolean mask of valid discrete actions for the current state.

        Shape: (NUM_EFFECT_ACTIONS,) -> (8,)
        True indicates action is valid; False indicates action must be masked out.
        """
        mask = torch.ones(NUM_EFFECT_ACTIONS, dtype=torch.bool, device=self.device)
        chain_len = len(self.current_chain)

        # 1. STOP Action constraints:
        # Cannot STOP before min_chain_length is satisfied
        if chain_len < self.min_chain_length:
            mask[STOP_ACTION_IDX] = False

        # 2. Maximum Chain Length constraints:
        # If max_chain_length reached, ONLY STOP is allowed
        if chain_len >= self.max_chain_length:
            mask[:STOP_ACTION_IDX] = False
            mask[STOP_ACTION_IDX] = True
            return mask

        # 3. Repeated Effects constraint:
        if not self.allow_repeated_effects:
            used_effects = {step["effect"] for step in self.current_chain}
            for i, effect_name in enumerate(SUPPORTED_EFFECTS):
                if effect_name in used_effects:
                    mask[i] = False

        # If all effects were masked, STOP must be available
        if not mask.any():
            mask[STOP_ACTION_IDX] = True

        return mask

    def step(
        self,
        discrete_action: int | torch.Tensor,
        continuous_params: torch.Tensor | list[float] | None = None,
    ) -> tuple[torch.Tensor, float, bool, bool, dict[str, Any]]:
        """Execute one sequential action in the environment.

        Parameters
        ----------
        discrete_action : int | torch.Tensor
            Index of chosen action in [0, 7].
        continuous_params : torch.Tensor | list[float], optional
            Normalized parameter values in [0, 1] for the selected effect.

        Returns
        -------
        obs : torch.Tensor
            Next state observation.
        reward : float
            Scalar step reward.
        terminated : bool
            True if episode ended naturally (e.g. STOP chosen or target reached).
        truncated : bool
            True if maximum allowed steps exceeded.
        info : dict[str, Any]
            Diagnostic information dictionary.
        """
        if isinstance(discrete_action, torch.Tensor):
            action_idx = int(discrete_action.item())
        else:
            action_idx = int(discrete_action)

        prev_loss = self.current_loss
        self.current_step += 1

        # Check action validity against current mask
        valid_mask = self.get_action_mask()
        action_was_masked = not bool(valid_mask[action_idx].item())

        # -------------------------------------------------------------
        # CASE 1: Terminal STOP Action
        # -------------------------------------------------------------
        chain_len = len(self.current_chain)
        if action_idx == STOP_ACTION_IDX or (action_was_masked and chain_len >= self.max_chain_length):
            terminated = True
            truncated = False

            # Calculate Terminal Reward
            # 1. Total improvement over dry baseline
            improvement_pct = max(0.0, (1.0 - self.current_loss / (self.initial_loss + 1e-8)) * 100.0)
            target_achieved = self.current_loss <= self.target_threshold_loss

            reward = 0.0
            if target_achieved:
                reward += 3.0  # Target achievement bonus
            if len(self.current_chain) == 0:
                reward -= 1.0  # Penalty for doing nothing when min_length expected
            else:
                reward += (improvement_pct / 100.0) * 2.0

            obs = self._get_observation()
            info = {
                "action": "STOP",
                "step": self.current_step,
                "chain_length": len(self.current_chain),
                "prev_loss": prev_loss,
                "current_loss": self.current_loss,
                "delta_loss": 0.0,
                "reward": reward,
                "improvement_pct": improvement_pct,
                "target_achieved": target_achieved,
                "action_mask": self.get_action_mask(),
            }
            return obs, reward, terminated, truncated, info

        # -------------------------------------------------------------
        # CASE 2: Add Effect to Chain
        # -------------------------------------------------------------
        chosen_effect = EFFECT_ACTIONS[action_idx]
        param_names = PARAM_SCHEMA[chosen_effect]
        k_params = len(param_names)

        # Parse and clamp continuous parameters
        if continuous_params is None:
            chosen_p = torch.full((k_params,), 0.5, device=self.device)
        else:
            if not isinstance(continuous_params, torch.Tensor):
                chosen_p = torch.tensor(continuous_params, dtype=torch.float32, device=self.device)
            else:
                chosen_p = continuous_params.to(self.device)
            chosen_p = torch.clamp(chosen_p[:k_params], 0.0, 1.0)

        # Map to parameter dictionary
        param_dict = {
            p_name: float(chosen_p[i].item())
            for i, p_name in enumerate(param_names)
        }

        # Append to chain in exact sequential order
        self.current_chain.append({
            "effect": chosen_effect,
            "parameters": param_dict,
        })

        # Render through DAPN
        self._render_current_sound()

        # Compute new spectral distance loss
        with torch.no_grad():
            try:
                loss_tensor = self.loss_fn(self.current_audio, self.target_audio)
                new_loss = float(loss_tensor.item())
            except Exception as e:
                logger.warning("Loss computation failed: %s", e)
                new_loss = prev_loss + 1.0

        if not math.isfinite(new_loss):
            new_loss = prev_loss + 1.0

        self.current_loss = new_loss

        # Calculate Step Reward
        # delta_loss > 0 means improvement
        delta_loss = prev_loss - new_loss
        reward = (delta_loss * 10.0) - self.step_penalty

        if action_was_masked:
            reward -= 0.5  # Soft penalty if agent chose an invalid effect

        # Check termination conditions
        target_achieved = self.current_loss <= self.target_threshold_loss
        terminated = bool(target_achieved)
        truncated = bool(len(self.current_chain) >= self.max_chain_length or self.current_step >= self.max_chain_length + 2)

        if target_achieved:
            reward += 3.0

        obs = self._get_observation()
        info = {
            "action": chosen_effect,
            "step": self.current_step,
            "chain_length": len(self.current_chain),
            "prev_loss": prev_loss,
            "current_loss": new_loss,
            "delta_loss": delta_loss,
            "reward": reward,
            "improvement_pct": max(0.0, (1.0 - new_loss / (self.initial_loss + 1e-8)) * 100.0),
            "target_achieved": target_achieved,
            "action_mask": self.get_action_mask(),
        }
        return obs, reward, terminated, truncated, info

    def _render_current_sound(self) -> None:
        """Render audio using current sequential chain through DAPN."""
        if not self.current_chain:
            self.current_audio = self.dry_audio.clone()
            return

        chain_dict = {"chain": self.current_chain}
        with torch.no_grad():
            try:
                rendered = self.dapn.process(self.dry_audio, chain_dict)
                if not torch.isfinite(rendered).all():
                    logger.warning("Rendered audio contains non-finite values; falling back to dry.")
                    self.current_audio = self.dry_audio.clone()
                else:
                    self.current_audio = rendered
            except Exception as e:
                logger.warning("DAPN processing failed: %s; falling back to dry audio.", e)
                self.current_audio = self.dry_audio.clone()

    def _get_observation(self) -> torch.Tensor:
        """Construct sequential state representation (shape: 148)."""
        with torch.no_grad():
            # 1. Acoustic distance embedding (64 features)
            n_fft = 512
            win = torch.hann_window(n_fft, device=self.device)
            c_stft = torch.stft(
                self.current_audio.squeeze(0), n_fft=n_fft, hop_length=256, window=win, return_complex=True
            ).abs()
            t_stft = torch.stft(
                self.target_audio.squeeze(0), n_fft=n_fft, hop_length=256, window=win, return_complex=True
            ).abs()

            diff = (t_stft - c_stft).mean(dim=-1).squeeze(0)
            if diff.shape[-1] >= self.acoustic_dim:
                acoustic_feats = F.adaptive_avg_pool1d(diff.unsqueeze(0).unsqueeze(0), self.acoustic_dim).squeeze()
            else:
                acoustic_feats = F.pad(diff, (0, self.acoustic_dim - diff.shape[-1]))

            # 2. Ordered Chain History (8 slots * 8 classes = 64 features)
            # Each slot is a one-hot vector indicating [EMPTY (0), EQ (1), ..., Flanger (7)]
            num_classes = len(SUPPORTED_EFFECTS) + 1  # 8
            history_matrix = torch.zeros((self.history_slots, num_classes), device=self.device)
            for slot_idx, step in enumerate(self.current_chain[:self.history_slots]):
                eff_name = step["effect"]
                class_idx = SUPPORTED_EFFECTS.index(eff_name) + 1  # 1..7 (0 is EMPTY)
                history_matrix[slot_idx, class_idx] = 1.0

            # Empty slots have index 0 = 1.0
            for slot_idx in range(len(self.current_chain), self.history_slots):
                history_matrix[slot_idx, 0] = 1.0

            history_feats = history_matrix.flatten()  # 64

            # 3. Scalar Step & Progress Metrics (4 features)
            step_norm = float(self.current_step) / float(self.max_chain_length)
            chain_len_norm = float(len(self.current_chain)) / float(self.max_chain_length)
            loss_norm = min(1.0, self.current_loss / 5.0)
            improvement_ratio = max(0.0, 1.0 - (self.current_loss / (self.initial_loss + 1e-8)))
            metrics_feats = torch.tensor([step_norm, chain_len_norm, loss_norm, improvement_ratio], device=self.device)

            # 4. Concatenate: Acoustic (64) + History (64) + Metrics (4) + PTA Taste (16) = 148 dims
            obs = torch.cat([acoustic_feats, history_feats, metrics_feats, self.pta_taste_vector], dim=0)
            obs = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
            return obs

    def get_current_chain_summary(self) -> dict[str, Any]:
        """Returns clean dictionary summary of the accumulated sequential chain."""
        return {
            "chain": copy.deepcopy(self.current_chain),
            "loss": round(self.current_loss, 5),
            "num_effects": len(self.current_chain),
            "sequence": [step["effect"] for step in self.current_chain],
        }
