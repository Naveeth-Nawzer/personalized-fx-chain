"""
rl/benchmark_comparison.py
==========================
Benchmark comparison across three synthesis approaches on held-out test audio:
  1. Baseline A: M2 Inverse FX Warm-Start Only (zero-shot deep neural prediction)
  2. Baseline B: Direct Gradient Descent using Adam on Differentiable DSP (continuous parameter optimization)
  3. Proposed Method: M2 Warm-Start + PPO Agent (hybrid discrete structural & continuous refinement)

Ensures fair, identical evaluation conditions on the same audio pairs, durations, and metrics.

Usage:
------
python -m rl.benchmark_comparison --checkpoint checkpoints/ppo/best_ppo_model.pth --num-samples 5
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from common.losses import MultiResolutionSTFTLoss
from dapn.dapn import DAPN
from dataset.dapn_dataset import (
    DAPNSyntheticDataset,
    PARAM_OFFSETS,
    PARAM_SCHEMA,
    SUPPORTED_EFFECTS,
    TOTAL_NUM_PARAMS,
)
from models.inverse_fx_net import InverseFXNet
from rl.dapn_env import AudioFXEnv
from rl.ppo_agent import PPOAgent
from rl.grrl_env import GRRLEnv
from rl.grrl_agent import GRRLAgent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_proposed_grrl(
    env: GRRLEnv,
    agent: GRRLAgent,
    dry_audio: torch.Tensor,
    target_audio: torch.Tensor,
    max_steps: int = 8,
    use_warm_start: bool = False,
) -> dict[str, Any]:
    """Proposed Dynamic GRRL-Agent: Sequential graph generation with explicit STOP."""
    t0 = time.time()
    obs, info = env.reset(dry_audio, target_audio, use_warm_start=use_warm_start)

    done = False
    step_num = 0

    while not done and step_num < max_steps:
        step_num += 1
        mask = env.get_action_mask()
        disc_act, cont_act, _, _ = agent.select_action(obs, action_mask=mask, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(disc_act, continuous_params=cont_act)
        done = terminated or truncated

    runtime = time.time() - t0
    summary = env.get_current_chain_summary()

    return {
        "final_loss": round(env.current_loss, 5),
        "steps": step_num,
        "runtime_sec": round(runtime, 4),
        "active_effects": summary["sequence"],
        "num_effects": len(summary["sequence"]),
        "chain": summary["chain"],
    }


def run_baseline_a_warm_start(
    predictor: InverseFXNet,
    dapn: DAPN,
    loss_fn: MultiResolutionSTFTLoss,
    dry_audio: torch.Tensor,
    target_audio: torch.Tensor,
) -> dict[str, Any]:
    """Baseline A: Zero-shot Inverse FX Net prediction without further refinement."""
    t0 = time.time()
    with torch.no_grad():
        logits, pred_params = predictor(dry_audio, target_audio)
        presence = (torch.sigmoid(logits)[0] >= 0.5).float()
        params = torch.clamp(pred_params[0], 0.0, 1.0)

        chain_steps = []
        for i, eff in enumerate(SUPPORTED_EFFECTS):
            if presence[i] > 0.5:
                start_i, end_i = PARAM_OFFSETS[eff]
                p_names = PARAM_SCHEMA[eff]
                p_dict = {p_names[k]: float(params[start_i + k].item()) for k in range(len(p_names))}
                chain_steps.append({"effect": eff, "parameters": p_dict})

        chain_dict = {"chain": chain_steps}
        if chain_steps:
            pred_audio = dapn.process(dry_audio, chain_dict)
        else:
            pred_audio = dry_audio.clone()

        final_loss = float(loss_fn(pred_audio, target_audio).item())

    runtime = time.time() - t0
    active_effects = [SUPPORTED_EFFECTS[i] for i in range(len(SUPPORTED_EFFECTS)) if presence[i] > 0.5]

    return {
        "final_loss": round(final_loss, 5),
        "steps": 0,
        "runtime_sec": round(runtime, 4),
        "active_effects": active_effects,
        "num_effects": len(active_effects),
        "presence_tensor": presence,
        "params_tensor": params,
    }


def run_baseline_b_adam(
    dapn: DAPN,
    loss_fn: MultiResolutionSTFTLoss,
    dry_audio: torch.Tensor,
    target_audio: torch.Tensor,
    init_presence: torch.Tensor,
    init_params: torch.Tensor,
    num_steps: int = 10,
    lr: float = 0.05,
) -> dict[str, Any]:
    """Baseline B: Continuous parameter optimization using Adam and Autograd through DAPN."""
    t0 = time.time()

    # Active effects chain structure is fixed based on warm-start presence
    active_indices = [i for i in range(len(SUPPORTED_EFFECTS)) if init_presence[i] > 0.5]
    if not active_indices:
        # If no effect was predicted active, return warm start result directly
        with torch.no_grad():
            final_loss = float(loss_fn(dry_audio, target_audio).item())
        return {
            "final_loss": round(final_loss, 5),
            "steps": num_steps,
            "runtime_sec": round(time.time() - t0, 4),
            "active_effects": [],
            "num_effects": 0,
        }

    # Prepare learnable continuous parameter overrides
    learnable_params = torch.nn.Parameter(init_params.clone().detach())
    optimizer = torch.optim.Adam([learnable_params], lr=lr)

    final_loss = float("inf")

    for _ in range(num_steps):
        optimizer.zero_grad()
        clamped_p = torch.clamp(learnable_params, 0.0, 1.0)

        chain_steps = []
        override = {}
        for c_idx, e_idx in enumerate(active_indices):
            eff = SUPPORTED_EFFECTS[e_idx]
            start_i, end_i = PARAM_OFFSETS[eff]
            p_names = PARAM_SCHEMA[eff]

            p_override = {}
            p_static = {}
            for k, p_name in enumerate(p_names):
                p_val = clamped_p[start_i + k]
                p_override[p_name] = p_val
                p_static[p_name] = float(p_val.detach().item())

            override[c_idx] = p_override
            chain_steps.append({"effect": eff, "parameters": p_static})

        chain_dict = {"chain": chain_steps}
        pred_audio = dapn.process(dry_audio, chain_dict, param_override=override)
        loss = loss_fn(pred_audio, target_audio)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())

    runtime = time.time() - t0
    active_effects = [SUPPORTED_EFFECTS[i] for i in active_indices]

    return {
        "final_loss": round(final_loss, 5),
        "steps": num_steps,
        "runtime_sec": round(runtime, 4),
        "active_effects": active_effects,
        "num_effects": len(active_effects),
    }


def run_proposed_ppo(
    env: AudioFXEnv,
    agent: PPOAgent,
    dry_audio: torch.Tensor,
    target_audio: torch.Tensor,
    max_steps: int = 10,
) -> dict[str, Any]:
    """Proposed Method: Warm-start initialization + PPO policy refinement."""
    t0 = time.time()
    obs, info = env.reset(dry_audio, target_audio, use_warm_start=True)

    done = False
    step_num = 0

    while not done and step_num < max_steps:
        step_num += 1
        action_env, _, _, _, _ = agent.select_action(obs, deterministic=True)
        obs, reward, terminated, truncated, step_info = env.step(action_env)
        done = terminated or truncated

    runtime = time.time() - t0
    summary = env.get_current_chain_summary()
    active_effects = [item["effect"] for item in summary["chain"]]

    return {
        "final_loss": round(env.current_loss, 5),
        "steps": step_num,
        "runtime_sec": round(runtime, 4),
        "active_effects": active_effects,
        "num_effects": len(active_effects),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark M3 PPO / GRRL against Baseline A (Warm-Start) and Baseline B (Adam)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained fixed-order PPO baseline checkpoint")
    parser.add_argument("--grrl-checkpoint", type=str, default=None, help="Path to trained Dynamic GRRL-Agent checkpoint")
    parser.add_argument("--predictor-checkpoint", type=str, default="checkpoints/best_model.pth", help="InverseFXNet checkpoint")
    parser.add_argument("--val-metadata", type=str, default="dataset/synthetic/val/metadata.jsonl", help="Validation metadata JSONL")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of held-out samples to benchmark")
    parser.add_argument("--steps", type=int, default=10, help="Number of refinement steps per method")
    parser.add_argument("--output-json", type=str, default="benchmark_results.json", help="Path to save results JSON")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device (cpu, cuda, mps)")
    args = parser.parse_args()

    # Load shared modules
    loss_fn = MultiResolutionSTFTLoss()
    dapn = DAPN()

    predictor = InverseFXNet().to(args.device)
    ckpt = torch.load(args.predictor_checkpoint, map_location=args.device)
    predictor.load_state_dict(ckpt["model_state_dict"])
    predictor.eval()

    # Fixed-order PPO agent (Baseline C)
    env_fixed = None
    agent_fixed = None
    if args.checkpoint and Path(args.checkpoint).exists():
        env_fixed = AudioFXEnv(
            predictor_checkpoint=args.predictor_checkpoint,
            device=args.device,
            max_steps_per_episode=args.steps,
        )
        agent_fixed = PPOAgent(
            obs_dim=env_fixed.obs_dim,
            discrete_dim=env_fixed.num_effects,
            continuous_dim=env_fixed.num_params,
            device=args.device,
        )
        agent_fixed.load(args.checkpoint)

    # Dynamic GRRL agent (Proposed)
    env_grrl = None
    agent_grrl = None
    if args.grrl_checkpoint and Path(args.grrl_checkpoint).exists():
        env_grrl = GRRLEnv(
            predictor_checkpoint=args.predictor_checkpoint,
            device=args.device,
            max_chain_length=8,
        )
        agent_grrl = GRRLAgent(
            obs_dim=env_grrl.obs_dim,
            num_actions=env_grrl.num_actions,
            device=args.device,
        )
        agent_grrl.load(args.grrl_checkpoint)

    # Load evaluation samples
    dataset = DAPNSyntheticDataset(args.val_metadata, max_samples=args.num_samples)
    logger.info("Loaded %d held-out validation samples from %s", len(dataset), args.val_metadata)

    results = []

    print("\n" + "=" * 105)
    print(f" BENCHMARK COMPARISON: Audio Effects Chain Synthesis (N={len(dataset)} Samples)")
    print("=" * 105)

    header = f"{'Sample ID':<14} | {'Metric':<16} | {'Baseline A (M2)':<16} | {'Baseline B (Adam)':<18}"
    if agent_fixed is not None:
        header += f" | {'Baseline C (PPO)':<16}"
    if agent_grrl is not None:
        header += f" | {'Proposed (GRRL)':<16}"
    print(header)
    print("-" * 105)

    for i in range(len(dataset)):
        sample = dataset[i]
        sample_id = sample["sample_id"]
        dry = sample["dry_audio"].unsqueeze(0) if sample["dry_audio"].ndim == 2 else sample["dry_audio"]
        wet = sample["wet_audio"].unsqueeze(0) if sample["wet_audio"].ndim == 2 else sample["wet_audio"]

        with torch.no_grad():
            initial_baseline_loss = float(loss_fn(dry, wet).item())

        # 1. Baseline A: M2 Warm-Start only
        res_a = run_baseline_a_warm_start(predictor, dapn, loss_fn, dry, wet)
        pct_a = max(0.0, (1.0 - res_a["final_loss"] / (initial_baseline_loss + 1e-8)) * 100.0)

        # 2. Baseline B: Adam Gradient Descent
        res_b = run_baseline_b_adam(
            dapn=dapn,
            loss_fn=loss_fn,
            dry_audio=dry,
            target_audio=wet,
            init_presence=res_a["presence_tensor"],
            init_params=res_a["params_tensor"],
            num_steps=args.steps,
        )
        pct_b = max(0.0, (1.0 - res_b["final_loss"] / (initial_baseline_loss + 1e-8)) * 100.0)

        # 3. Baseline C: Fixed-order PPO (if available)
        res_fixed = None
        pct_fixed = 0.0
        if agent_fixed is not None and env_fixed is not None:
            res_fixed = run_proposed_ppo(
                env=env_fixed,
                agent=agent_fixed,
                dry_audio=dry,
                target_audio=wet,
                max_steps=args.steps,
            )
            pct_fixed = max(0.0, (1.0 - res_fixed["final_loss"] / (initial_baseline_loss + 1e-8)) * 100.0)

        # 4. Proposed Method: Dynamic GRRL (if available)
        res_grrl = None
        pct_grrl = 0.0
        if agent_grrl is not None and env_grrl is not None:
            res_grrl = run_proposed_grrl(
                env=env_grrl,
                agent=agent_grrl,
                dry_audio=dry,
                target_audio=wet,
                max_steps=8,
            )
            pct_grrl = max(0.0, (1.0 - res_grrl["final_loss"] / (initial_baseline_loss + 1e-8)) * 100.0)

        sample_summary = {
            "sample_id": sample_id,
            "dry_baseline_loss": round(initial_baseline_loss, 5),
            "baseline_a_warm_start": {
                "loss": res_a["final_loss"],
                "improvement_pct": round(pct_a, 2),
                "active_effects": res_a["active_effects"],
                "runtime_sec": res_a["runtime_sec"],
            },
            "baseline_b_adam": {
                "loss": res_b["final_loss"],
                "improvement_pct": round(pct_b, 2),
                "active_effects": res_b["active_effects"],
                "runtime_sec": res_b["runtime_sec"],
                "steps": res_b["steps"],
            },
        }
        if res_fixed is not None:
            sample_summary["baseline_c_fixed_ppo"] = {
                "loss": res_fixed["final_loss"],
                "improvement_pct": round(pct_fixed, 2),
                "active_effects": res_fixed["active_effects"],
                "runtime_sec": res_fixed["runtime_sec"],
                "steps": res_fixed["steps"],
            }
        if res_grrl is not None:
            sample_summary["proposed_grrl"] = {
                "loss": res_grrl["final_loss"],
                "improvement_pct": round(pct_grrl, 2),
                "active_effects": res_grrl["active_effects"],
                "runtime_sec": res_grrl["runtime_sec"],
                "steps": res_grrl["steps"],
            }

        results.append(sample_summary)

        row_loss = f"{sample_id:<14} | Final MR-STFT    | {res_a['final_loss']:<16.5f} | {res_b['final_loss']:<18.5f}"
        row_pct = f"{'':<14} | Improvement %    | {pct_a:<15.1f}% | {pct_b:<17.1f}%"
        row_eff = f"{'':<14} | Active Effects   | {res_a['num_effects']:<16d} | {res_b['num_effects']:<18d}"
        row_time = f"{'':<14} | Runtime (sec)    | {res_a['runtime_sec']:<16.3f} | {res_b['runtime_sec']:<18.3f}"

        if res_fixed is not None:
            row_loss += f" | {res_fixed['final_loss']:<16.5f}"
            row_pct += f" | {pct_fixed:<15.1f}%"
            row_eff += f" | {res_fixed['num_effects']:<16d}"
            row_time += f" | {res_fixed['runtime_sec']:<16.3f}"
        if res_grrl is not None:
            row_loss += f" | {res_grrl['final_loss']:<16.5f}"
            row_pct += f" | {pct_grrl:<15.1f}%"
            row_eff += f" | {res_grrl['num_effects']:<16d}"
            row_time += f" | {res_grrl['runtime_sec']:<16.3f}"

        print(row_loss)
        print(row_pct)
        print(row_eff)
        print(row_time)
        print("-" * 105)

    # Compute overall averages
    avg_loss_a = sum(r["baseline_a_warm_start"]["loss"] for r in results) / len(results)
    avg_loss_b = sum(r["baseline_b_adam"]["loss"] for r in results) / len(results)
    avg_pct_a = sum(r["baseline_a_warm_start"]["improvement_pct"] for r in results) / len(results)
    avg_pct_b = sum(r["baseline_b_adam"]["improvement_pct"] for r in results) / len(results)

    avg_summary = {
        "num_samples": len(results),
        "avg_loss_baseline_a": round(avg_loss_a, 5),
        "avg_loss_baseline_b": round(avg_loss_b, 5),
        "avg_improvement_baseline_a_pct": round(avg_pct_a, 2),
        "avg_improvement_baseline_b_pct": round(avg_pct_b, 2),
    }

    avg_loss_str = f"{'AVERAGE':<14} | Final MR-STFT    | {avg_loss_a:<16.5f} | {avg_loss_b:<18.5f}"
    avg_pct_str = f"{'':<14} | Improvement %    | {avg_pct_a:<15.1f}% | {avg_pct_b:<17.1f}%"

    if any("baseline_c_fixed_ppo" in r for r in results):
        avg_loss_f = sum(r["baseline_c_fixed_ppo"]["loss"] for r in results) / len(results)
        avg_pct_f = sum(r["baseline_c_fixed_ppo"]["improvement_pct"] for r in results) / len(results)
        avg_summary["avg_loss_baseline_c_fixed_ppo"] = round(avg_loss_f, 5)
        avg_summary["avg_improvement_baseline_c_pct"] = round(avg_pct_f, 2)
        avg_loss_str += f" | {avg_loss_f:<16.5f}"
        avg_pct_str += f" | {avg_pct_f:<15.1f}%"

    if any("proposed_grrl" in r for r in results):
        avg_loss_g = sum(r["proposed_grrl"]["loss"] for r in results) / len(results)
        avg_pct_g = sum(r["proposed_grrl"]["improvement_pct"] for r in results) / len(results)
        avg_summary["avg_loss_proposed_grrl"] = round(avg_loss_g, 5)
        avg_summary["avg_improvement_proposed_grrl_pct"] = round(avg_pct_g, 2)
        avg_loss_str += f" | {avg_loss_g:<16.5f}"
        avg_pct_str += f" | {avg_pct_g:<15.1f}%"

    print("=" * 105)
    print(avg_loss_str)
    print(avg_pct_str)
    print("=" * 105)

    # Save output report
    out_file = Path(args.output_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": avg_summary,
                "samples": results,
            },
            f,
            indent=2,
        )
    logger.info("Saved complete benchmark comparison report to %s", out_file)



if __name__ == "__main__":
    main()
