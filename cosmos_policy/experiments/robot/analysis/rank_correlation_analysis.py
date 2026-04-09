# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment 0-A Analysis: Shallow-Full Rank Correlation.

Reads the JSON output from collect_candidate_data.py and computes:
1. Global Spearman rho between shallow and full value rankings
2. Breakdown by episode phase (early/mid/late)
3. Breakdown by task difficulty (easy/hard)
4. Breakdown by value range (low/mid/high)
5. Per-state rho distribution

Usage:
    python -m cosmos_policy.experiments.robot.analysis.rank_correlation_analysis \
        --input_path ../../adaptive-ttc-wam/experiments/0A_rank_correlation/candidate_data_libero_spatial_*.json \
        --output_dir ../../adaptive-ttc-wam/experiments/0A_rank_correlation
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def compute_spearman_rho(shallow_values, full_values):
    """Compute Spearman rank correlation between shallow and full value rankings."""
    if len(shallow_values) < 3:
        return np.nan, np.nan
    rho, p_value = stats.spearmanr(shallow_values, full_values)
    return rho, p_value


def compute_kendall_tau(shallow_values, full_values):
    """Compute Kendall tau rank correlation."""
    if len(shallow_values) < 3:
        return np.nan, np.nan
    tau, p_value = stats.kendalltau(shallow_values, full_values)
    return tau, p_value


def analyze_decision_point(dp):
    """Analyze a single decision point: extract shallow/full values and compute correlation."""
    candidates = dp["candidates"]
    if len(candidates) < 3:
        return None

    shallow_values = [c["shallow_value"] for c in candidates]
    # Exclude index 0 (= shallow_value) from full to prevent target leakage
    full_values = [float(np.mean(c["full_values"][1:])) for c in candidates]

    rho, rho_p = compute_spearman_rho(shallow_values, full_values)
    tau, tau_p = compute_kendall_tau(shallow_values, full_values)

    # Rank agreement: does shallow pick the same top-1 as full?
    shallow_best_idx = np.argmax(shallow_values)
    full_best_idx = np.argmax(full_values)
    top1_agreement = int(shallow_best_idx == full_best_idx)

    # Top-k agreement: do shallow top-3 overlap with full top-3?
    k = min(3, len(candidates))
    shallow_topk = set(np.argsort(shallow_values)[-k:])
    full_topk = set(np.argsort(full_values)[-k:])
    topk_overlap = len(shallow_topk & full_topk) / k

    # Mean full value (proxy for state difficulty — high value = easy state)
    mean_full_value = np.mean(full_values)

    # Split-half verifier stability: split the 14 remaining full values into two
    # halves per candidate and correlate the two half-means against each other.
    # This measures how noisy the "oracle" full verifier is.
    full_vs_full_rho = np.nan
    if len(candidates) >= 3:
        half_a_vals = []
        half_b_vals = []
        for c in candidates:
            remaining = c["full_values"][1:]  # 14 values after excluding index 0
            mid = len(remaining) // 2
            half_a_vals.append(float(np.mean(remaining[:mid])) if mid > 0 else np.nan)
            half_b_vals.append(float(np.mean(remaining[mid:])) if len(remaining) - mid > 0 else np.nan)
        if not any(np.isnan(half_a_vals)) and not any(np.isnan(half_b_vals)):
            fvf_rho, _ = stats.spearmanr(half_a_vals, half_b_vals)
            full_vs_full_rho = float(fvf_rho)

    return {
        "spearman_rho": rho,
        "spearman_p": rho_p,
        "kendall_tau": tau,
        "kendall_p": tau_p,
        "top1_agreement": top1_agreement,
        "topk_overlap": topk_overlap,
        "mean_full_value": mean_full_value,
        "full_vs_full_rho": full_vs_full_rho,
        "episode_phase": dp.get("episode_phase", "unknown"),
        "timestep": dp.get("timestep", -1),
        "num_candidates": len(candidates),
    }


def run_analysis(input_path, output_dir):
    """Run the full rank correlation analysis."""
    # Load data
    with open(input_path, "r") as f:
        data = json.load(f)

    config = data["config"]
    print(f"Loaded data: {config['task_suite']}, {config['num_candidates']} candidates, "
          f"{config['num_wm_rollouts_full']}WM x {config['num_value_preds_full']}V full verification")

    # Analyze all decision points
    all_analyses = []
    task_analyses = {}

    for task_data in data["tasks"]:
        task_id = task_data["task_id"]
        task_desc = task_data["task_description"]
        task_results = []

        for episode in task_data["episodes"]:
            for dp in episode["decision_points"]:
                result = analyze_decision_point(dp)
                if result is not None:
                    result["task_id"] = task_id
                    result["task_description"] = task_desc
                    result["episode_idx"] = episode["episode_idx"]
                    result["episode_success"] = episode["success"]
                    all_analyses.append(result)
                    task_results.append(result)

        task_analyses[task_id] = task_results

    if not all_analyses:
        print("ERROR: No valid decision points found!")
        return

    print(f"\nAnalyzed {len(all_analyses)} decision points across {len(data['tasks'])} tasks")

    # ==========================================
    # 1. Global statistics
    # ==========================================
    rhos = [a["spearman_rho"] for a in all_analyses if not np.isnan(a["spearman_rho"])]
    taus = [a["kendall_tau"] for a in all_analyses if not np.isnan(a["kendall_tau"])]
    top1s = [a["top1_agreement"] for a in all_analyses]
    topks = [a["topk_overlap"] for a in all_analyses]
    fvf_rhos = [a["full_vs_full_rho"] for a in all_analyses if not np.isnan(a["full_vs_full_rho"])]

    # Bootstrap 95% CI for mean Spearman rho (1000 resamples at decision-point level)
    rng = np.random.default_rng(42)
    rhos_arr = np.array(rhos)
    bootstrap_means = np.array([
        rng.choice(rhos_arr, size=len(rhos_arr), replace=True).mean()
        for _ in range(1000)
    ])
    rho_ci_low = float(np.percentile(bootstrap_means, 2.5))
    rho_ci_high = float(np.percentile(bootstrap_means, 97.5))

    global_stats = {
        "num_decision_points": len(all_analyses),
        "spearman_rho": {
            "mean": float(np.mean(rhos)),
            "median": float(np.median(rhos)),
            "std": float(np.std(rhos)),
            "min": float(np.min(rhos)),
            "max": float(np.max(rhos)),
            "q25": float(np.percentile(rhos, 25)),
            "q75": float(np.percentile(rhos, 75)),
            "bootstrap_ci_95": [rho_ci_low, rho_ci_high],
        },
        "kendall_tau": {
            "mean": float(np.mean(taus)),
            "median": float(np.median(taus)),
            "std": float(np.std(taus)),
        },
        "top1_agreement_rate": float(np.mean(top1s)),
        "topk_overlap_rate": float(np.mean(topks)),
        "full_vs_full_rho": {
            "mean": float(np.mean(fvf_rhos)) if fvf_rhos else None,
            "median": float(np.median(fvf_rhos)) if fvf_rhos else None,
            "std": float(np.std(fvf_rhos)) if fvf_rhos else None,
        },
    }

    print(f"\n{'='*60}")
    print(f"GLOBAL RANK CORRELATION RESULTS")
    print(f"{'='*60}")
    print(f"Spearman rho:  mean={global_stats['spearman_rho']['mean']:.3f} "
          f"median={global_stats['spearman_rho']['median']:.3f} "
          f"std={global_stats['spearman_rho']['std']:.3f} "
          f"95% CI=[{rho_ci_low:.3f}, {rho_ci_high:.3f}]")
    print(f"Kendall tau:   mean={global_stats['kendall_tau']['mean']:.3f} "
          f"median={global_stats['kendall_tau']['median']:.3f}")
    print(f"Top-1 agreement: {global_stats['top1_agreement_rate']:.1%}")
    print(f"Top-3 overlap:   {global_stats['topk_overlap_rate']:.1%}")
    if fvf_rhos:
        print(f"Full-vs-full rho (verifier stability): "
              f"mean={global_stats['full_vs_full_rho']['mean']:.3f} "
              f"std={global_stats['full_vs_full_rho']['std']:.3f}")

    # Go/No-Go decision
    mean_rho = global_stats["spearman_rho"]["mean"]
    if mean_rho >= 0.6:
        decision = "PROCEED"
        decision_msg = f"rho={mean_rho:.3f} >= 0.6 -> PROCEED with adaptive allocation"
    elif mean_rho >= 0.4:
        decision = "CAUTION"
        decision_msg = f"rho={mean_rho:.3f} in [0.4, 0.6) -> PROCEED WITH CAUTION, consider increasing shallow budget"
    else:
        decision = "REDESIGN"
        decision_msg = f"rho={mean_rho:.3f} < 0.4 -> STOP AND REDESIGN shallow verification strategy"

    print(f"\n*** GO/NO-GO DECISION: {decision} ***")
    print(f"    {decision_msg}")

    # ==========================================
    # 2. Breakdown by episode phase
    # ==========================================
    phase_stats = {}
    for phase in ["early", "mid", "late"]:
        phase_analyses = [a for a in all_analyses if a["episode_phase"] == phase]
        if phase_analyses:
            phase_rhos = [a["spearman_rho"] for a in phase_analyses if not np.isnan(a["spearman_rho"])]
            phase_stats[phase] = {
                "count": len(phase_analyses),
                "spearman_rho_mean": float(np.mean(phase_rhos)) if phase_rhos else None,
                "spearman_rho_std": float(np.std(phase_rhos)) if phase_rhos else None,
                "top1_agreement": float(np.mean([a["top1_agreement"] for a in phase_analyses])),
            }

    print(f"\n{'='*60}")
    print(f"BREAKDOWN BY EPISODE PHASE")
    print(f"{'='*60}")
    for phase, s in phase_stats.items():
        print(f"  {phase:5s}: rho={s['spearman_rho_mean']:.3f} +/- {s['spearman_rho_std']:.3f}, "
              f"top1={s['top1_agreement']:.1%}, n={s['count']}")

    # ==========================================
    # 3. Breakdown by value range (state difficulty proxy)
    # ==========================================
    mean_values = [a["mean_full_value"] for a in all_analyses]
    v_low = np.percentile(mean_values, 33)
    v_high = np.percentile(mean_values, 67)

    value_range_stats = {}
    for label, low, high in [("low", -1, v_low), ("mid", v_low, v_high), ("high", v_high, 2)]:
        group = [a for a in all_analyses if low <= a["mean_full_value"] < high]
        if group:
            group_rhos = [a["spearman_rho"] for a in group if not np.isnan(a["spearman_rho"])]
            value_range_stats[label] = {
                "count": len(group),
                "value_range": [float(low), float(high)],
                "spearman_rho_mean": float(np.mean(group_rhos)) if group_rhos else None,
                "spearman_rho_std": float(np.std(group_rhos)) if group_rhos else None,
                "top1_agreement": float(np.mean([a["top1_agreement"] for a in group])),
            }

    print(f"\n{'='*60}")
    print(f"BREAKDOWN BY VALUE RANGE (state difficulty)")
    print(f"{'='*60}")
    for label, s in value_range_stats.items():
        print(f"  {label:4s} (V in [{s['value_range'][0]:.2f}, {s['value_range'][1]:.2f})): "
              f"rho={s['spearman_rho_mean']:.3f} +/- {s['spearman_rho_std']:.3f}, "
              f"top1={s['top1_agreement']:.1%}, n={s['count']}")

    # ==========================================
    # 4. Per-task breakdown
    # ==========================================
    per_task_stats = {}
    print(f"\n{'='*60}")
    print(f"PER-TASK BREAKDOWN")
    print(f"{'='*60}")
    for task_id, task_results in task_analyses.items():
        if task_results:
            task_rhos = [a["spearman_rho"] for a in task_results if not np.isnan(a["spearman_rho"])]
            task_desc = task_results[0]["task_description"]
            per_task_stats[task_id] = {
                "task_description": task_desc,
                "count": len(task_results),
                "spearman_rho_mean": float(np.mean(task_rhos)) if task_rhos else None,
                "top1_agreement": float(np.mean([a["top1_agreement"] for a in task_results])),
            }
            print(f"  Task {task_id} ({task_desc[:40]}): "
                  f"rho={per_task_stats[task_id]['spearman_rho_mean']:.3f}, "
                  f"top1={per_task_stats[task_id]['top1_agreement']:.1%}, "
                  f"n={per_task_stats[task_id]['count']}")

    # ==========================================
    # 5. Save results
    # ==========================================
    results = {
        "config": config,
        "go_no_go_decision": decision,
        "go_no_go_message": decision_msg,
        "global": global_stats,
        "by_episode_phase": phase_stats,
        "by_value_range": value_range_stats,
        "by_task": per_task_stats,
        "per_state_rho_distribution": rhos,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rank_correlation_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 0-A: Shallow-Full Rank Correlation Analysis")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to candidate_data JSON from collect_candidate_data.py")
    parser.add_argument("--output_dir", type=str,
                        default="../../adaptive-ttc-wam/experiments/0A_rank_correlation",
                        help="Output directory for results")
    args = parser.parse_args()

    # Handle glob patterns
    if "*" in args.input_path:
        files = sorted(glob.glob(args.input_path))
        if not files:
            raise FileNotFoundError(f"No files matching: {args.input_path}")
        args.input_path = files[-1]  # Use most recent
        print(f"Using most recent file: {args.input_path}")

    run_analysis(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()
