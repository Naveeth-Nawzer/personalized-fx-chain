"""
rl/train_ppo.py
===============
Training script for M3 PPO Reinforcement Learning Agent on DAPN AudioFXEnv.

Usage:
------
python -m rl.train_ppo --total-timesteps 2000 --rollout-steps 64
python -m rl.train_ppo --total-timesteps 5000 --lr 3e-4 --checkpoint-dir checkpoints/ppo
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dataset.dapn_dataset import DAPNSyntheticDataset
from rl.dapn_env import AudioFXEnv
from rl.ppo_agent import PPOAgent
from rl.rollout_buffer import RolloutBuffer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_synthetic_pair(device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Generates synthetic dry and harmonic target tone pair if dataset is not present."""
    sr = 44100
    t = torch.linspace(0, 1.5, int(sr * 1.5), device=device)
    base_freq = random.choice([220.0, 330.0, 440.0, 550.0])
    dry = torch.sin(2 * 3.14159 * base_freq * t).unsqueeze(0)
    target = (
        torch.sin(2 * 3.14159 * base_freq * t)
        + 0.4 * torch.sin(2 * 3.14159 * base_freq * 2.0 * t)
        + 0.2 * torch.sin(2 * 3.14159 * base_freq * 3.0 * t)
    ).unsqueeze(0)
    return dry, target


def evaluate_policy(
    env: AudioFXEnv,
    agent: PPOAgent,
    val_dataset: DAPNSyntheticDataset | None,
    num_episodes: int = 5,
) -> dict[str, float]:
    """Evaluate current policy deterministically on validation episodes."""
    total_rewards = []
    initial_losses = []
    final_losses = []
    improvement_pcts = []

    for ep in range(num_episodes):
        if val_dataset is not None and len(val_dataset) > 0:
            sample = val_dataset[ep % len(val_dataset)]
            dry = sample["dry_audio"]
            target = sample["wet_audio"]
        else:
            dry, target = create_synthetic_pair()

        obs, info = env.reset(dry, target, use_warm_start=True)
        init_l = info["initial_loss"]
        ep_reward = 0.0
        done = False
        last_loss = info["current_loss"]

        while not done:
            action_env, _, _, _, _ = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = env.step(action_env)
            ep_reward += reward
            last_loss = step_info["current_loss"]
            done = terminated or truncated

        total_rewards.append(ep_reward)
        initial_losses.append(init_l)
        final_losses.append(last_loss)
        pct = max(0.0, (1.0 - last_loss / (init_l + 1e-8)) * 100.0)
        improvement_pcts.append(pct)

    return {
        "eval_mean_reward": sum(total_rewards) / max(1, len(total_rewards)),
        "eval_initial_loss": sum(initial_losses) / max(1, len(initial_losses)),
        "eval_final_loss": sum(final_losses) / max(1, len(final_losses)),
        "eval_improvement_pct": sum(improvement_pcts) / max(1, len(improvement_pcts)),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Initialize environment
    predictor_ckpt = "checkpoints/best_model.pth" if Path("checkpoints/best_model.pth").exists() else None
    env = AudioFXEnv(
        predictor_checkpoint=predictor_ckpt,
        device=args.device,
        max_steps_per_episode=args.max_episode_steps,
        sample_rate=44100,
        audio_duration_sec=2.0,
    )

    # Initialize datasets if available
    train_dataset: DAPNSyntheticDataset | None = None
    if Path(args.train_metadata).exists():
        train_dataset = DAPNSyntheticDataset(args.train_metadata, max_samples=args.max_train_samples)
        logger.info("Loaded training dataset: %d samples from %s", len(train_dataset), args.train_metadata)

    val_dataset: DAPNSyntheticDataset | None = None
    if Path(args.val_metadata).exists():
        val_dataset = DAPNSyntheticDataset(args.val_metadata, max_samples=args.max_val_samples)
        logger.info("Loaded validation dataset: %d samples from %s", len(val_dataset), args.val_metadata)

    # Initialize PPO agent and Rollout buffer
    agent = PPOAgent(
        obs_dim=env.obs_dim,
        discrete_dim=env.num_effects,
        continuous_dim=env.num_params,
        lr=args.lr,
        clip_eps=args.clip_eps,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
    )

    buffer = RolloutBuffer(
        buffer_size=args.rollout_steps,
        obs_dim=env.obs_dim,
        discrete_dim=env.num_effects,
        continuous_dim=env.num_params,
        device=args.device,
    )

    # Resume from checkpoint if requested
    if args.resume and Path(args.resume).exists():
        agent.load(args.resume)
        logger.info("Resumed policy from %s", args.resume)

    logger.info("=" * 70)
    logger.info(" Starting M3 PPO Policy Training for Audio FX Synthesis")
    logger.info(" Total Timesteps: %d | Rollout Size: %d | Epochs: %d | Batch Size: %d",
                args.total_timesteps, args.rollout_steps, args.epochs, args.batch_size)
    logger.info("=" * 70)

    # Episode loop initialization
    step_count = 0
    episode_count = 0
    best_eval_loss = float("inf")
    metrics_log_path = checkpoint_dir / "training_metrics.jsonl"

    # Helper to get next training pair
    train_idx = 0
    def get_train_pair():
        nonlocal train_idx
        if train_dataset is not None and len(train_dataset) > 0:
            sample = train_dataset[train_idx % len(train_dataset)]
            train_idx += 1
            return sample["dry_audio"], sample["wet_audio"]
        return create_synthetic_pair()

    dry, target = get_train_pair()
    obs, info = env.reset(dry, target, use_warm_start=True)
    ep_reward = 0.0
    ep_start_loss = info["initial_loss"]

    start_time = time.time()

    while step_count < args.total_timesteps:
        buffer.reset()

        # 1. Collect rollout steps
        while not buffer.is_full and step_count < args.total_timesteps:
            action_env, disc_act, cont_act, log_prob, val = agent.select_action(obs, deterministic=False)
            next_obs, reward, terminated, truncated, step_info = env.step(action_env)

            done = terminated or truncated
            buffer.add(
                obs=obs,
                discrete_action=disc_act,
                continuous_action=cont_act,
                reward=reward,
                value=val,
                log_prob=log_prob,
                done=done,
            )

            step_count += 1
            ep_reward += reward

            if done:
                episode_count += 1
                logger.debug(
                    "Episode %d finished: reward=%.2f, start_loss=%.4f, final_loss=%.4f, steps=%d",
                    episode_count, ep_reward, ep_start_loss, step_info["current_loss"], step_info["step"]
                )
                dry, target = get_train_pair()
                obs, info = env.reset(dry, target, use_warm_start=True)
                ep_reward = 0.0
                ep_start_loss = info["initial_loss"]
            else:
                obs = next_obs

        # 2. Bootstrap value for incomplete episode at buffer end
        with torch.no_grad():
            _, _, _, _, last_val = agent.actor_critic.get_action(obs, deterministic=True)
            last_val_float = float(last_val.item())

        # 3. Compute returns and GAE
        buffer.compute_returns_and_advantages(
            last_value=last_val_float,
            last_done=done,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
        )

        # 4. PPO update step
        train_metrics = agent.train_step(
            buffer=buffer,
            batch_size=args.batch_size,
            epochs=args.epochs,
        )

        # 5. Periodic evaluation and logging
        elapsed = time.time() - start_time
        fps = step_count / max(0.1, elapsed)

        logger.info(
            "Step %5d / %5d | Policy Loss: %7.4f | Value Loss: %7.4f | Entropy: %6.2f | KL: %6.4f | Speed: %.1f steps/s",
            step_count, args.total_timesteps,
            train_metrics.get("policy_loss", 0.0),
            train_metrics.get("value_loss", 0.0),
            train_metrics.get("entropy", 0.0),
            train_metrics.get("approx_kl", 0.0),
            fps,
        )

        # Validation check
        if step_count % args.eval_interval == 0 or step_count >= args.total_timesteps:
            eval_metrics = evaluate_policy(env, agent, val_dataset, num_episodes=args.eval_episodes)
            logger.info(
                "--- [Validation @ Step %d] Reward: %.2f | Loss: %.4f (Init: %.4f) | Improvement: %.1f%% ---",
                step_count,
                eval_metrics["eval_mean_reward"],
                eval_metrics["eval_final_loss"],
                eval_metrics["eval_initial_loss"],
                eval_metrics["eval_improvement_pct"],
            )

            # Record metrics to JSONL log
            log_entry = {
                "step": step_count,
                "elapsed_sec": round(elapsed, 2),
                **train_metrics,
                **eval_metrics,
            }
            with open(metrics_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Checkpoint best policy
            if eval_metrics["eval_final_loss"] < best_eval_loss:
                best_eval_loss = eval_metrics["eval_final_loss"]
                best_ckpt_path = checkpoint_dir / "best_ppo_model.pth"
                agent.save(best_ckpt_path)
                logger.info("Saved new BEST policy checkpoint to %s (Loss: %.4f)", best_ckpt_path, best_eval_loss)

    # Final checkpoint save
    final_ckpt_path = checkpoint_dir / "final_ppo_model.pth"
    agent.save(final_ckpt_path)
    logger.info("=" * 70)
    logger.info(" Training Complete! Final checkpoint saved to: %s", final_ckpt_path)
    logger.info("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train M3 PPO Agent for Audio FX Chain Synthesis")
    parser.add_argument("--total-timesteps", type=int, default=2000, help="Total environment interaction timesteps")
    parser.add_argument("--rollout-steps", type=int, default=64, help="Timesteps collected per PPO rollout buffer")
    parser.add_argument("--epochs", type=int, default=4, help="PPO update epochs per rollout")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size for PPO updates")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda parameter")
    parser.add_argument("--clip-eps", type=float, default=0.20, help="PPO clipping epsilon")
    parser.add_argument("--value-coef", type=float, default=0.50, help="Value function loss coefficient")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.50, help="Max gradient norm for clipping")
    parser.add_argument("--max-episode-steps", type=int, default=10, help="Max environment steps per episode")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation frequency in timesteps")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/ppo", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--train-metadata", type=str, default="dataset/synthetic/train/metadata.jsonl", help="Train metadata path")
    parser.add_argument("--val-metadata", type=str, default="dataset/synthetic/val/metadata.jsonl", help="Val metadata path")
    parser.add_argument("--max-train-samples", type=int, default=100, help="Max train samples to load")
    parser.add_argument("--max-val-samples", type=int, default=20, help="Max val samples to load")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu, cuda, mps)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
