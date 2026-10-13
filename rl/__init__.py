"""
rl/__init__.py
==============
Reinforcement Learning Component for Collaborative Audio Effects Chain Synthesis.
"""

from rl.dapn_env import AudioFXEnv
from rl.ppo_agent import ActorCritic, PPOAgent
from rl.rollout_buffer import RolloutBuffer
from rl.grrl_env import GRRLEnv, EFFECT_ACTIONS, NUM_EFFECT_ACTIONS, STOP_ACTION_IDX
from rl.grrl_agent import GRRLActorCritic, GRRLAgent
from rl.grrl_buffer import GRRLRolloutBuffer

__all__ = [
    # Fixed-order Baseline
    "AudioFXEnv",
    "ActorCritic",
    "PPOAgent",
    "RolloutBuffer",
    # Dynamic Sequential GRRL-Agent
    "GRRLEnv",
    "GRRLActorCritic",
    "GRRLAgent",
    "GRRLRolloutBuffer",
    "EFFECT_ACTIONS",
    "NUM_EFFECT_ACTIONS",
    "STOP_ACTION_IDX",
]

