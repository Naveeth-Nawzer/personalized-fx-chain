"""
rl/evaluate_ppo.py
==================
Inference and decision logging script for the trained M3 PPO Reinforcement Learning Agent.

Proves the RL agent makes active, state-dependent decisions across discrete effect toggles
and continuous knob parameters, recording a full decision audit trail.

Usage:
------
python -m rl.evaluate_ppo --checkpoint checkpoints/ppo/best_ppo_model.pth --output-log evaluation_log.jsonl
python -m rl.evaluate_ppo --checkpoint checkpoints/ppo/best_ppo_model.pth --deterministic --max-steps 10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from audio_io import load_audio
from dataset.dapn_dataset import DAPNSyntheticDataset, PARAM_SCHEMA, SUPPORTED_EFFECTS
from rl.dapn_env import AudioFXEnv
from rl.ppo_agent import PPOAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_single_pair(
    env: AudioFXEnv,
    agent: PPOAgent,
    dry_audio: torch.Tensor,
    target_audio: torch.Tensor,
    sample_id: str = "sample_000001",
    deterministic: bool = True,
    max_steps: int = 10,
) -> list[dict[str, Any]]:
    """Run full episode evaluation on a single audio pair and record detailed step decisions."""
    obs, info = env.reset(dry_audio, target_audio, use_warm_start=True)

    decision_log = []
    done = False
    step_num = 0

    logger.info("=" * 70)
    logger.info(" Evaluating Sample: %s | Mode: %s", sample_id, "Deterministic" if deterministic else "Stochastic")
    logger.info(" Initial Baseline Loss: %.5f | Active Effects: %d", info["initial_loss"], info["num_active_effects"])
    logger.info("=" * 70)

    while not done and step_num < max_steps:
        step_num += 1
        loss_before = env.current_loss
        prev_presence = env.current_presence.clone()
        prev_params = env.current_params.clone()

        # Policy forward inference
        action_env, disc_act, cont_act, log_prob, val = agent.select_action(obs, deterministic=deterministic)

        # Environment transition
        obs, reward, terminated, truncated, step_info = env.step(action_env)
        loss_after = step_info["current_loss"]
        done = terminated or truncated

        # Analyze changes made by the action
        active_effects_now = [SUPPORTED_EFFECTS[i] for i in range(len(SUPPORTED_EFFECTS)) if disc_act[i] > 0.5]
        toggled_on = [SUPPORTED_EFFECTS[i] for i in range(len(SUPPORTED_EFFECTS)) if disc_act[i] > 0.5 and prev_presence[i] <= 0.5]
        toggled_off = [SUPPORTED_EFFECTS[i] for i in range(len(SUPPORTED_EFFECTS)) if disc_act[i] <= 0.5 and prev_presence[i] > 0.5]

        param_delta_norm = float(torch.norm(cont_act).item())
        delta_loss = loss_before - loss_after

        log_record = {
            "sample_id": sample_id,
            "step": step_num,
            "obs_summary": {
                "obs_mean": round(float(obs.mean().item()), 4),
                "obs_std": round(float(obs.std().item()), 4),
            },
            "discrete_action_raw": [round(float(v), 2) for v in disc_act.tolist()],
            "active_effects": active_effects_now,
            "toggled_on": toggled_on,
            "toggled_off": toggled_off,
            "continuous_action_norm": round(param_delta_norm, 4),
            "loss_before": round(loss_before, 5),
            "loss_after": round(loss_after, 5),
            "loss_reduction": round(delta_loss, 5),
            "reward": round(reward, 4),
            "value_estimate": round(val, 4),
            "log_prob": round(log_prob, 4),
            "terminated": terminated,
            "truncated": truncated,
        }
        decision_log.append(log_record)

        logger.info(
            "Step %2d | Loss: %.5f -> %.5f (d=%.5f) | Rew: %6.2f | Active: %s | Toggled: +%s -%s",
            step_num, loss_before, loss_after, delta_loss, reward, active_effects_now, toggled_on, toggled_off
        )

    init_l = info["initial_loss"]
    improvement_pct = max(0.0, (1.0 - env.current_loss / (init_l + 1e-8)) * 100.0)
    logger.info("=" * 70)
    logger.info(" Episode Finished at Step %d | Total Improvement: %.2f%% | Final Loss: %.5f",
                step_num, improvement_pct, env.current_loss)
    logger.info("=" * 70)

    return decision_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Trained PPO Policy on AudioFXEnv with Decision Logging")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained PPO checkpoint")
    parser.add_argument("--dry-audio", type=str, default=None, help="Path to single dry WAV audio file")
    parser.add_argument("--target-audio", type=str, default=None, help="Path to single target WAV audio file")
    parser.add_argument("--dataset-metadata", type=str, default="dataset/synthetic/val/metadata.jsonl", help="Validation metadata JSONL")
    parser.add_argument("--max-samples", type=int, default=3, help="Number of audio pairs to evaluate")
    parser.add_argument("--max-steps", type=int, default=10, help="Max steps per episode")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic action selection")
    parser.add_argument("--stochastic", action="store_false", dest="deterministic", help="Stochastic action selection")
    parser.add_argument("--output-log", type=str, default="decision_log.jsonl", help="Path to output JSONL log file")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu, cuda, mps)")
    args = parser.parse_args()

    # Load environment and agent
    predictor_ckpt = "checkpoints/best_model.pth" if Path("checkpoints/best_model.pth").exists() else None
    env = AudioFXEnv(
        predictor_checkpoint=predictor_ckpt,
        device=args.device,
        max_steps_per_episode=args.max_steps,
    )

    agent = PPOAgent(
        obs_dim=env.obs_dim,
        discrete_dim=env.num_effects,
        continuous_dim=env.num_params,
        device=args.device,
    )
    agent.load(args.checkpoint)

    all_logs = []

    # Case 1: Direct file inputs
    if args.dry_audio and args.target_audio:
        dry, sr_dry = load_audio(args.dry_audio)
        target, sr_target = load_audio(args.target_audio)
        logs = evaluate_single_pair(
            env=env,
            agent=agent,
            dry_audio=dry,
            target_audio=target,
            sample_id="custom_input",
            deterministic=args.deterministic,
            max_steps=args.max_steps,
        )
        all_logs.extend(logs)
    # Case 2: Validation dataset
    elif Path(args.dataset_metadata).exists():
        dataset = DAPNSyntheticDataset(args.dataset_metadata, max_samples=args.max_samples)
        for i in range(len(dataset)):
            sample = dataset[i]
            logs = evaluate_single_pair(
                env=env,
                agent=agent,
                dry_audio=sample["dry_audio"],
                target_audio=sample["wet_audio"],
                sample_id=sample["sample_id"],
                deterministic=args.deterministic,
                max_steps=args.max_steps,
            )
            all_logs.extend(logs)
    # Case 3: Synthetic test tone fallback
    else:
        logger.warning("No audio files or dataset found; evaluating on synthetic sine wave.")
        t = torch.linspace(0, 1.0, 44100)
        dry = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
        target = (torch.sin(2 * 3.14159 * 440 * t) + 0.3 * torch.sin(2 * 3.14159 * 880 * t)).unsqueeze(0)
        logs = evaluate_single_pair(
            env=env,
            agent=agent,
            dry_audio=dry,
            target_audio=target,
            sample_id="synthetic_tone",
            deterministic=args.deterministic,
            max_steps=args.max_steps,
        )
        all_logs.extend(logs)

    # Save output log
    out_path = Path(args.output_log)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        # Write CSV
        if all_logs:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(all_logs[0].keys()))
                writer.writeheader()
                writer.writerows(all_logs)
    else:
        # Write JSONL
        with open(out_path, "w", encoding="utf-8") as f:
            for entry in all_logs:
                f.write(json.dumps(entry) + "\n")

    logger.info("Saved %d decision logs to: %s", len(all_logs), out_path)


if __name__ == "__main__":
    main()
