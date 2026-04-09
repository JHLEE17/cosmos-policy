# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment 0-Zero-B (Phase 1): Value-Accuracy Correlation Analysis.

Uses existing candidate data + episode success labels to test whether
higher value predictions correlate with actual task success.

Analyses:
1. Episode-level: Do episodes with higher mean values succeed more?
2. Step-level: At each decision point, does the selected action's value
   predict subsequent episode success?
3. Calibration: Are value predictions calibrated against actual success rates?
4. Simulated selection strategies: If we HAD selected by value,
   would success rate improve? (counterfactual from observational data)

Usage:
    python -m cosmos_policy.experiments.robot.analysis.value_accuracy_analysis \
        --input_path ../../adaptive-ttc-wam/experiments/0A_rank_correlation/candidate_data_libero_spatial_merged.json \
        --output_dir ../../adaptive-ttc-wam/experiments/0Zero_value_validity
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def compute_episode_level_correlation(data):
    """
    Episode-level analysis: correlation between mean value and episode success.

    For each episode, compute the average value across all decision points
    and all candidates' full_value_mean. Then correlate with success.
    """
    episode_records = []

    for task_data in data["tasks"]:
        task_id = task_data["task_id"]
        for episode in task_data["episodes"]:
            if not episode["decision_points"]:
                continue

            # Aggregate values across all decision points in this episode
            all_candidate_means = []
            best_candidate_values = []
            worst_candidate_values = []

            for dp in episode["decision_points"]:
                candidates = dp["candidates"]
                if len(candidates) < 2:
                    continue

                # Full values (excluding shallow to avoid leakage)
                full_means = [float(np.mean(c["full_values"][1:])) for c in candidates]
                shallow_values = [c["shallow_value"] for c in candidates]

                all_candidate_means.extend(full_means)
                best_candidate_values.append(max(full_means))
                worst_candidate_values.append(min(full_means))

            if not all_candidate_means:
                continue

            episode_records.append({
                "task_id": task_id,
                "episode_idx": episode["episode_idx"],
                "success": episode["success"],
                "mean_value_all": float(np.mean(all_candidate_means)),
                "mean_best_value": float(np.mean(best_candidate_values)),
                "mean_worst_value": float(np.mean(worst_candidate_values)),
                "value_spread": float(np.mean(best_candidate_values) - np.mean(worst_candidate_values)),
                "num_decision_points": len(episode["decision_points"]),
            })

    return episode_records


def compute_step_level_correlation(data):
    """
    Step-level analysis: at each decision point, do higher-value candidates
    appear more in successful episodes?

    Also: rank candidates by shallow vs full and check if the "best" choice
    correlates with episode success.
    """
    step_records = []

    for task_data in data["tasks"]:
        task_id = task_data["task_id"]
        for episode in task_data["episodes"]:
            for dp in episode["decision_points"]:
                candidates = dp["candidates"]
                if len(candidates) < 2:
                    continue

                full_means = [float(np.mean(c["full_values"][1:])) for c in candidates]
                shallow_values = [c["shallow_value"] for c in candidates]

                # Best candidate by different strategies
                best_full_idx = np.argmax(full_means)
                best_shallow_idx = np.argmax(shallow_values)
                random_idx = 0  # Candidate with seed=base_seed (effectively "default")

                step_records.append({
                    "task_id": task_id,
                    "episode_idx": episode["episode_idx"],
                    "episode_success": episode["success"],
                    "timestep": dp.get("timestep", -1),
                    "episode_phase": dp.get("episode_phase", "unknown"),
                    "best_full_value": float(full_means[best_full_idx]),
                    "best_shallow_value": float(shallow_values[best_shallow_idx]),
                    "default_value": float(full_means[random_idx]),
                    "mean_value": float(np.mean(full_means)),
                    "max_value": float(max(full_means)),
                    "min_value": float(min(full_means)),
                    "value_range": float(max(full_means) - min(full_means)),
                    "shallow_full_agree": int(best_full_idx == best_shallow_idx),
                    "num_candidates": len(candidates),
                })

    return step_records


def compute_value_calibration(step_records, n_bins=5):
    """
    Calibration analysis: bin decision points by value, compute actual success rate per bin.

    If value is well-calibrated, higher value bins should have higher success rates.
    """
    # Use best_full_value as the value measure
    values = [r["best_full_value"] for r in step_records]
    successes = [r["episode_success"] for r in step_records]

    # Bin by quantile
    quantiles = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    bins = []

    for i in range(n_bins):
        lo = quantiles[i]
        hi = quantiles[i + 1]
        if i == n_bins - 1:
            mask = [lo <= v <= hi for v in values]
        else:
            mask = [lo <= v < hi for v in values]

        bin_successes = [s for s, m in zip(successes, mask) if m]
        bin_values = [v for v, m in zip(values, mask) if m]

        if bin_successes:
            bins.append({
                "bin_idx": i,
                "value_range": [float(lo), float(hi)],
                "mean_value": float(np.mean(bin_values)),
                "n": len(bin_successes),
                "success_rate": float(np.mean(bin_successes)),
                "n_success": int(sum(bin_successes)),
            })

    return bins


def simulate_selection_strategies(data):
    """
    Counterfactual analysis: for each episode, simulate what would happen
    if we selected the best/worst/random candidate at each step.

    Since we can't actually re-run episodes, we use a proxy:
    - Compare the VALUE of the selected candidate across strategies
    - This tells us IF value matters, WHICH strategy captures it best

    Also compute: per-episode "best-worst gap" to measure if there's even
    a meaningful choice to make.
    """
    episode_strategy_values = []

    for task_data in data["tasks"]:
        task_id = task_data["task_id"]
        for episode in task_data["episodes"]:
            if not episode["decision_points"]:
                continue

            strategy_values = {
                "greedy_full": [],
                "greedy_shallow": [],
                "worst_full": [],
                "random": [],
                "mean_all": [],
            }
            value_gaps = []

            for dp in episode["decision_points"]:
                candidates = dp["candidates"]
                if len(candidates) < 2:
                    continue

                full_means = [float(np.mean(c["full_values"][1:])) for c in candidates]
                shallow_values = [c["shallow_value"] for c in candidates]

                strategy_values["greedy_full"].append(max(full_means))
                strategy_values["greedy_shallow"].append(full_means[np.argmax(shallow_values)])
                strategy_values["worst_full"].append(min(full_means))
                strategy_values["random"].append(full_means[0])
                strategy_values["mean_all"].append(float(np.mean(full_means)))

                value_gaps.append(max(full_means) - min(full_means))

            if not value_gaps:
                continue

            episode_strategy_values.append({
                "task_id": task_id,
                "episode_idx": episode["episode_idx"],
                "success": episode["success"],
                "mean_greedy_full_value": float(np.mean(strategy_values["greedy_full"])),
                "mean_greedy_shallow_value": float(np.mean(strategy_values["greedy_shallow"])),
                "mean_worst_value": float(np.mean(strategy_values["worst_full"])),
                "mean_random_value": float(np.mean(strategy_values["random"])),
                "mean_all_value": float(np.mean(strategy_values["mean_all"])),
                "mean_value_gap": float(np.mean(value_gaps)),
                "max_value_gap": float(max(value_gaps)),
            })

    return episode_strategy_values


def run_analysis(input_path, output_dir):
    """Run the full value-accuracy correlation analysis."""
    with open(input_path, "r") as f:
        data = json.load(f)

    config = data["config"]
    print(f"Loaded: {config['task_suite']}, {config['num_candidates']} candidates")

    # ==========================================
    # 1. Episode-level correlation
    # ==========================================
    print(f"\n{'='*60}")
    print(f"EPISODE-LEVEL VALUE-SUCCESS CORRELATION")
    print(f"{'='*60}")

    episode_records = compute_episode_level_correlation(data)
    n_episodes = len(episode_records)
    n_success = sum(1 for r in episode_records if r["success"])

    print(f"Episodes: {n_episodes}, Success: {n_success} ({n_success/n_episodes:.1%})")

    # Point-biserial correlation: success (binary) vs mean value
    values_arr = np.array([r["mean_value_all"] for r in episode_records])
    success_arr = np.array([int(r["success"]) for r in episode_records])

    if len(set(success_arr)) > 1:
        pb_corr, pb_pval = stats.pointbiserialr(success_arr, values_arr)
        print(f"Point-biserial r(success, mean_value): {pb_corr:.4f}, p={pb_pval:.4f}")

        # Also with best value
        best_values = np.array([r["mean_best_value"] for r in episode_records])
        pb_best_corr, pb_best_pval = stats.pointbiserialr(success_arr, best_values)
        print(f"Point-biserial r(success, best_value): {pb_best_corr:.4f}, p={pb_best_pval:.4f}")

        # Mann-Whitney U: are values higher in successful episodes?
        success_vals = [r["mean_value_all"] for r in episode_records if r["success"]]
        fail_vals = [r["mean_value_all"] for r in episode_records if not r["success"]]
        if success_vals and fail_vals:
            u_stat, u_pval = stats.mannwhitneyu(success_vals, fail_vals, alternative="greater")
            auc = u_stat / (len(success_vals) * len(fail_vals))
            print(f"Mann-Whitney U test (success > fail): AUC={auc:.4f}, p={u_pval:.4f}")
            print(f"  Success episodes mean value: {np.mean(success_vals):.4f} ± {np.std(success_vals):.4f}")
            print(f"  Failure episodes mean value: {np.mean(fail_vals):.4f} ± {np.std(fail_vals):.4f}")
    else:
        pb_corr, pb_pval = np.nan, np.nan
        pb_best_corr, pb_best_pval = np.nan, np.nan
        auc = np.nan
        print("WARNING: All episodes have same outcome — cannot compute correlation")

    # ==========================================
    # 2. Step-level analysis
    # ==========================================
    print(f"\n{'='*60}")
    print(f"STEP-LEVEL VALUE-SUCCESS CORRELATION")
    print(f"{'='*60}")

    step_records = compute_step_level_correlation(data)
    print(f"Decision points: {len(step_records)}")

    # Step-level point-biserial
    step_values = np.array([r["best_full_value"] for r in step_records])
    step_success = np.array([int(r["episode_success"]) for r in step_records])

    if len(set(step_success)) > 1:
        step_pb, step_pb_p = stats.pointbiserialr(step_success, step_values)
        print(f"Point-biserial r(success, best_value): {step_pb:.4f}, p={step_pb_p:.4f}")

        # By phase
        for phase in ["early", "mid", "late"]:
            phase_steps = [r for r in step_records if r["episode_phase"] == phase]
            if len(phase_steps) > 10:
                pv = np.array([r["best_full_value"] for r in phase_steps])
                ps = np.array([int(r["episode_success"]) for r in phase_steps])
                if len(set(ps)) > 1:
                    ppb, ppb_p = stats.pointbiserialr(ps, pv)
                    print(f"  {phase:5s}: r={ppb:.4f}, p={ppb_p:.4f}, n={len(phase_steps)}")
    else:
        step_pb, step_pb_p = np.nan, np.nan

    # ==========================================
    # 3. Calibration
    # ==========================================
    print(f"\n{'='*60}")
    print(f"VALUE CALIBRATION (does higher value → higher success rate?)")
    print(f"{'='*60}")

    calibration_bins = compute_value_calibration(step_records, n_bins=5)
    is_monotone = True
    prev_rate = -1

    for b in calibration_bins:
        arrow = "↑" if b["success_rate"] > prev_rate else "↓" if b["success_rate"] < prev_rate else "="
        if b["success_rate"] < prev_rate:
            is_monotone = False
        print(f"  Bin {b['bin_idx']}: value=[{b['value_range'][0]:.3f}, {b['value_range'][1]:.3f}], "
              f"success={b['success_rate']:.1%} ({b['n_success']}/{b['n']})"
              f"  {arrow}")
        prev_rate = b["success_rate"]

    # Spearman correlation on binned data
    if len(calibration_bins) >= 3:
        bin_values = [b["mean_value"] for b in calibration_bins]
        bin_rates = [b["success_rate"] for b in calibration_bins]
        cal_rho, cal_p = stats.spearmanr(bin_values, bin_rates)
        print(f"\nCalibration Spearman ρ: {cal_rho:.4f} (p={cal_p:.4f})")
        print(f"Monotone increasing? {'YES' if is_monotone else 'NO'}")
    else:
        cal_rho, cal_p = np.nan, np.nan

    # ==========================================
    # 4. Strategy simulation
    # ==========================================
    print(f"\n{'='*60}")
    print(f"SIMULATED SELECTION STRATEGIES")
    print(f"{'='*60}")

    strategy_records = simulate_selection_strategies(data)

    # For each strategy, compute correlation between mean strategy value and success
    strategy_results = {}
    for strategy_key in ["mean_greedy_full_value", "mean_greedy_shallow_value",
                         "mean_worst_value", "mean_random_value"]:
        strat_vals = np.array([r[strategy_key] for r in strategy_records])
        strat_success = np.array([int(r["success"]) for r in strategy_records])

        strategy_name = strategy_key.replace("mean_", "").replace("_value", "")
        if len(set(strat_success)) > 1:
            s_pb, s_pb_p = stats.pointbiserialr(strat_success, strat_vals)
            # AUC
            succ_v = strat_vals[strat_success == 1]
            fail_v = strat_vals[strat_success == 0]
            if len(succ_v) > 0 and len(fail_v) > 0:
                u, _ = stats.mannwhitneyu(succ_v, fail_v, alternative="greater")
                s_auc = u / (len(succ_v) * len(fail_v))
            else:
                s_auc = np.nan
        else:
            s_pb, s_pb_p, s_auc = np.nan, np.nan, np.nan

        strategy_results[strategy_name] = {
            "pb_corr": float(s_pb) if not np.isnan(s_pb) else None,
            "pb_pval": float(s_pb_p) if not np.isnan(s_pb_p) else None,
            "auc": float(s_auc) if not np.isnan(s_auc) else None,
            "mean_value_success": float(np.mean(strat_vals[strat_success == 1])) if sum(strat_success) > 0 else None,
            "mean_value_failure": float(np.mean(strat_vals[strat_success == 0])) if sum(strat_success == 0) > 0 else None,
        }
        print(f"  {strategy_name:20s}: AUC={strategy_results[strategy_name]['auc']:.4f}, "
              f"r={strategy_results[strategy_name]['pb_corr']:.4f}, "
              f"p={strategy_results[strategy_name]['pb_pval']:.4f}")

    # Value gap analysis
    value_gaps = [r["mean_value_gap"] for r in strategy_records]
    print(f"\n  Mean best-worst value gap per episode: {np.mean(value_gaps):.4f} ± {np.std(value_gaps):.4f}")
    print(f"  Max gap observed: {max(r['max_value_gap'] for r in strategy_records):.4f}")

    # ==========================================
    # 5. Summary verdict
    # ==========================================
    print(f"\n{'='*60}")
    print(f"SUMMARY VERDICT")
    print(f"{'='*60}")

    episode_auc = auc if not np.isnan(auc) else 0.5
    if episode_auc > 0.65:
        verdict = "VALUE PREDICTIVE — higher value meaningfully predicts episode success"
    elif episode_auc > 0.55:
        verdict = "WEAKLY PREDICTIVE — some value-success relationship exists but it's weak"
    elif episode_auc > 0.45:
        verdict = "NOT PREDICTIVE — value predictions do not distinguish success from failure"
    else:
        verdict = "INVERTED — higher value actually predicts FAILURE (value function is miscalibrated)"

    print(f"  Episode-level AUC: {episode_auc:.4f}")
    print(f"  Verdict: {verdict}")

    # ==========================================
    # 6. Save results
    # ==========================================
    results = {
        "config": config,
        "experiment": "0-Zero-B Phase 1: Value-Accuracy Correlation (observational)",
        "num_episodes": n_episodes,
        "num_success": n_success,
        "success_rate": n_success / n_episodes if n_episodes > 0 else 0,
        "episode_level": {
            "pb_corr_mean_value": float(pb_corr) if not np.isnan(pb_corr) else None,
            "pb_pval_mean_value": float(pb_pval) if not np.isnan(pb_pval) else None,
            "pb_corr_best_value": float(pb_best_corr) if not np.isnan(pb_best_corr) else None,
            "pb_pval_best_value": float(pb_best_pval) if not np.isnan(pb_best_pval) else None,
            "auc_success_vs_failure": float(auc) if not np.isnan(auc) else None,
            "success_mean_value": float(np.mean([r["mean_value_all"] for r in episode_records if r["success"]])) if n_success > 0 else None,
            "failure_mean_value": float(np.mean([r["mean_value_all"] for r in episode_records if not r["success"]])) if n_success < n_episodes else None,
        },
        "step_level": {
            "pb_corr": float(step_pb) if not np.isnan(step_pb) else None,
            "pb_pval": float(step_pb_p) if not np.isnan(step_pb_p) else None,
        },
        "calibration": {
            "bins": calibration_bins,
            "spearman_rho": float(cal_rho) if not np.isnan(cal_rho) else None,
            "spearman_pval": float(cal_p) if not np.isnan(cal_p) else None,
            "is_monotone": is_monotone,
        },
        "strategy_simulation": strategy_results,
        "value_gap": {
            "mean": float(np.mean(value_gaps)),
            "std": float(np.std(value_gaps)),
            "max": float(max(r["max_value_gap"] for r in strategy_records)),
        },
        "verdict": verdict,
        "episode_records": episode_records,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "value_accuracy_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 0-Zero-B: Value-Accuracy Correlation Analysis")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to candidate_data JSON")
    parser.add_argument("--output_dir", type=str,
                        default="../../adaptive-ttc-wam/experiments/0Zero_value_validity",
                        help="Output directory")
    args = parser.parse_args()

    if "*" in args.input_path:
        files = sorted(glob.glob(args.input_path))
        if not files:
            raise FileNotFoundError(f"No files matching: {args.input_path}")
        args.input_path = files[-1]
        print(f"Using most recent file: {args.input_path}")

    run_analysis(args.input_path, args.output_dir)


if __name__ == "__main__":
    main()
