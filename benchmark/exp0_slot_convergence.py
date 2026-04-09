"""
Experiment 0: Per-Slot Denoising Convergence Analysis

Motivation: In Cosmos Policy, action/proprio/value slots contain low-dimensional
vectors repeated to fill 16x28x28 latent frames, while camera slots contain actual
image content. We hypothesize these slot types converge at different rates during
denoising, which would justify asymmetric compute allocation.

Method:
1. For each step count N in [1..max_steps], run denoising with N steps
2. Use max_steps output as reference
3. Measure per-slot MSE and action extraction error vs reference
4. Same seed = same initial noise, so differences come purely from step count

Output: JSON + console table showing convergence curves per slot type.
"""

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# LIBERO slot layout
SLOT_NAMES = {
    0: "blank", 1: "curr_proprio", 2: "curr_wrist_cam", 3: "curr_primary_cam",
    4: "action", 5: "fut_proprio", 6: "fut_wrist_cam", 7: "fut_primary_cam", 8: "value",
}
SLOT_TYPES = {
    0: "blank", 1: "proprio", 2: "camera", 3: "camera",
    4: "action", 5: "proprio", 6: "camera", 7: "camera", 8: "value",
}
TARGET_INDICES = [4, 5, 6, 7, 8]  # indices that are generation targets (not conditioning)


def get_libero_config():
    return {
        "action_dim": 7, "proprio_dim": 9, "chunk_size": 16,
        "config": "cosmos_predict2_2b_480p_libero__inference_only",
        "ckpt_path": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        "t5_emb_path": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
        "dataset_stats_path": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json",
        "task_label": "pick up the black bowl on the stove and place it on the plate",
    }


def build_cfg(sc, num_steps):
    return SimpleNamespace(
        suite="libero", model_family="cosmos", config=sc["config"],
        ckpt_path=sc["ckpt_path"], config_file="cosmos_policy/config/config.py",
        use_third_person_image=True, num_third_person_images=1,
        use_wrist_image=True, num_wrist_images=1,
        use_proprio=True, flip_images=True,
        use_jpeg_compression=False, trained_with_image_aug=True,
        use_variance_scale=False, normalize_proprio=True, unnormalize_actions=False,
        chunk_size=sc["chunk_size"], num_open_loop_steps=sc["chunk_size"],
        ar_future_prediction=False, ar_value_prediction=False, ar_qvalue_prediction=False,
        num_denoising_steps_action=num_steps,
        t5_text_embeddings_path=sc["t5_emb_path"],
        dataset_stats_path=sc["dataset_stats_path"],
    )


def create_obs():
    s = 224
    return {
        "wrist_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
        "primary_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
        "proprio": np.random.randn(9).astype(np.float64),
    }


def compute_per_slot_metrics(pred_latent, ref_latent, chunk_size=16, action_dim=7):
    """Compute per-slot MSE between pred and reference latent."""
    B, C, T, H, W = ref_latent.shape
    metrics = {}

    for t_idx in TARGET_INDICES:
        slot_name = SLOT_NAMES[t_idx]
        slot_type = SLOT_TYPES[t_idx]

        ref_slot = ref_latent[:, :, t_idx, :, :]
        pred_slot = pred_latent[:, :, t_idx, :, :]

        mse = ((pred_slot - ref_slot) ** 2).mean().item()

        ref_flat = ref_slot.reshape(B, -1).float()
        pred_flat = pred_slot.reshape(B, -1).float()
        cos_sim = torch.nn.functional.cosine_similarity(ref_flat, pred_flat, dim=1).mean().item()

        metrics[slot_name] = {"mse": mse, "cosine_similarity": cos_sim, "slot_type": slot_type}

        # Action-level extraction error
        if slot_type == "action":
            from cosmos_policy.experiments.robot.cosmos_utils import extract_action_chunk_from_latent_sequence
            action_indices = torch.tensor([t_idx], dtype=torch.int64, device=ref_latent.device).expand(B)
            ref_action = extract_action_chunk_from_latent_sequence(ref_latent, (chunk_size, action_dim), action_indices)
            pred_action = extract_action_chunk_from_latent_sequence(pred_latent, (chunk_size, action_dim), action_indices)
            metrics[slot_name]["action_mse"] = ((pred_action - ref_action) ** 2).mean().item()
            metrics[slot_name]["action_max_error"] = (pred_action - ref_action).abs().max().item()

        # Value extraction error
        if slot_type == "value":
            ref_val = ref_slot.reshape(B, -1).float().mean(dim=1)
            pred_val = pred_slot.reshape(B, -1).float().mean(dim=1)
            metrics[slot_name]["value_abs_error"] = (pred_val - ref_val).abs().mean().item()

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Slot convergence analysis")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum denoising steps (reference)")
    parser.add_argument("--num-seeds", type=int, default=5, help="Number of seeds to average over")
    parser.add_argument("--output-dir", default="benchmark/results/slot_convergence")
    args = parser.parse_args()

    sc = get_libero_config()

    print(f"{'='*60}")
    print(f"  Exp 0: Per-Slot Denoising Convergence Analysis")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Max steps (reference): {args.max_steps}")
    print(f"  Seeds: {args.num_seeds}")
    print(f"{'='*60}")

    # Load model once
    cfg = build_cfg(sc, args.max_steps)
    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model, get_action, load_dataset_stats, init_t5_text_embeddings_cache,
    )
    model, config = get_model(cfg)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)

    # Warmup
    print("\nWarmup (3 iterations)...")
    obs = create_obs()
    for i in range(3):
        with torch.inference_mode():
            get_action(cfg, model, dataset_stats, obs, sc["task_label"],
                       seed=i, num_denoising_steps_action=5,
                       generate_future_state_and_value_in_parallel=False)
    print("Warmup done.\n")

    # Collect results across seeds
    all_seed_metrics = []  # list of dicts: {step_N: {slot_name: metrics}}

    for seed_idx in range(args.num_seeds):
        seed = 42 + seed_idx
        print(f"--- Seed {seed} ({seed_idx+1}/{args.num_seeds}) ---")

        # Fix observation per seed
        np.random.seed(seed)
        obs = create_obs()

        latents_by_steps = {}  # {N: latent_tensor}

        with torch.inference_mode():
            for n_steps in range(1, args.max_steps + 1):
                result = get_action(
                    cfg, model, dataset_stats, obs, sc["task_label"],
                    seed=seed, num_denoising_steps_action=n_steps,
                    generate_future_state_and_value_in_parallel=False,
                )
                latents_by_steps[n_steps] = result["generated_latent"].clone()
                print(f"  Step {n_steps:>2d}: latent captured  shape={result['generated_latent'].shape}")

        # Reference = max_steps output
        ref_latent = latents_by_steps[args.max_steps]

        seed_metrics = {}
        for n_steps in range(1, args.max_steps):
            seed_metrics[n_steps] = compute_per_slot_metrics(
                latents_by_steps[n_steps], ref_latent,
                chunk_size=sc["chunk_size"], action_dim=sc["action_dim"],
            )

        all_seed_metrics.append(seed_metrics)

    # Average across seeds
    print(f"\n{'='*90}")
    print(f"  Convergence Summary: Latent MSE vs {args.max_steps}-step reference (averaged over {args.num_seeds} seeds)")
    print(f"{'='*90}")

    averaged = {}
    for n_steps in range(1, args.max_steps):
        averaged[n_steps] = {}
        for slot_name in all_seed_metrics[0][1].keys():
            slot_metrics = {}
            for metric_key in ["mse", "cosine_similarity", "action_mse", "action_max_error", "value_abs_error"]:
                vals = [sm[n_steps][slot_name][metric_key]
                        for sm in all_seed_metrics if metric_key in sm[n_steps][slot_name]]
                if vals:
                    slot_metrics[metric_key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            slot_metrics["slot_type"] = all_seed_metrics[0][n_steps][slot_name]["slot_type"]
            averaged[n_steps][slot_name] = slot_metrics

    # Print MSE by slot type
    print(f"\n{'Step':>5s} | {'Action':>12s} | {'Camera(avg)':>12s} | {'Proprio(avg)':>12s} | {'Value':>12s} | {'Action CosSim':>14s}")
    print(f"{'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*14}")

    for n_steps in range(1, args.max_steps):
        type_mses = {"action": [], "camera": [], "proprio": [], "value": []}
        for slot_name, m in averaged[n_steps].items():
            st = m["slot_type"]
            if st in type_mses and "mse" in m:
                type_mses[st].append(m["mse"]["mean"])

        a = np.mean(type_mses["action"]) if type_mses["action"] else 0
        c = np.mean(type_mses["camera"]) if type_mses["camera"] else 0
        p = np.mean(type_mses["proprio"]) if type_mses["proprio"] else 0
        v = np.mean(type_mses["value"]) if type_mses["value"] else 0
        a_cos = averaged[n_steps]["action"].get("cosine_similarity", {}).get("mean", 0)

        print(f"{n_steps:>5d} | {a:>12.6f} | {c:>12.6f} | {p:>12.6f} | {v:>12.6f} | {a_cos:>14.8f}")

    # Print action extraction error
    print(f"\n{'Step':>5s} | {'ActionExtr MSE':>14s} | {'ActionExtr MaxE':>16s}")
    print(f"{'-'*5}-+-{'-'*14}-+-{'-'*16}")
    for n_steps in range(1, args.max_steps):
        a_mse = averaged[n_steps]["action"].get("action_mse", {}).get("mean", 0)
        a_max = averaged[n_steps]["action"].get("action_max_error", {}).get("mean", 0)
        print(f"{n_steps:>5d} | {a_mse:>14.8f} | {a_max:>16.8f}")

    # Compute convergence ratio: how much faster does action converge vs camera?
    print(f"\n{'='*60}")
    print(f"  Convergence Rate Analysis")
    print(f"{'='*60}")
    # Find step where each slot type reaches 95% cosine similarity
    for slot_type in ["action", "camera", "proprio", "value"]:
        for n_steps in range(1, args.max_steps):
            cos_vals = []
            for slot_name, m in averaged[n_steps].items():
                if m["slot_type"] == slot_type and "cosine_similarity" in m:
                    cos_vals.append(m["cosine_similarity"]["mean"])
            avg_cos = np.mean(cos_vals) if cos_vals else 0
            if avg_cos >= 0.95:
                print(f"  {slot_type:>8s}: reaches 95% cosine sim at step {n_steps}")
                break
        else:
            print(f"  {slot_type:>8s}: does NOT reach 95% cosine sim within {args.max_steps-1} steps")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "slot_convergence_results.json")
    output_data = {
        "metadata": {
            "gpu": torch.cuda.get_device_name(0),
            "max_steps": args.max_steps,
            "num_seeds": args.num_seeds,
            "suite": "libero",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "slot_layout": {str(k): v for k, v in SLOT_NAMES.items()},
            "slot_types": {str(k): v for k, v in SLOT_TYPES.items()},
            "target_indices": TARGET_INDICES,
        },
        "averaged_results": {str(k): v for k, v in averaged.items()},
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
