# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment 0-Zero-A (Phase 1): Value Function Stability Analysis.

Uses existing candidate data (3WM × 5V = 15 values per candidate) to measure:
1. ICC (Intra-class Correlation) — signal-to-noise ratio for candidate discrimination
2. CV (Coefficient of Variation) — relative spread of value predictions
3. Variance decomposition — between-candidate vs within-candidate vs between-WM variance
4. Denoising step effect (if applicable)

Data layout (rollout-major order):
  full_values[0:5]   = WM rollout 0, value predictions 0-4
  full_values[5:10]  = WM rollout 1, value predictions 0-4
  full_values[10:15] = WM rollout 2, value predictions 0-4
  shallow_value = full_values[0]

Usage:
    python -m cosmos_policy.experiments.robot.analysis.value_stability_analysis \
        --input_path ../../adaptive-ttc-wam/experiments/0A_rank_correlation/candidate_data_libero_spatial_merged.json \
        --output_dir ../../adaptive-ttc-wam/experiments/0Zero_value_validity
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def compute_icc_oneway(values_matrix):
    """
    Compute ICC(1,1) — one-way random effects model.

    values_matrix: shape (n_candidates, n_measurements)
    Each row = one candidate, each column = one value prediction.

    ICC = (MS_between - MS_within) / (MS_between + (k-1) * MS_within)

    Returns ICC value and interpretation.
    """
    n, k = values_matrix.shape
    if n < 2 or k < 2:
        return np.nan, np.nan, np.nan

    # Grand mean
    grand_mean = np.mean(values_matrix)

    # Between-candidate: variance of row means around grand mean
    row_means = np.mean(values_matrix, axis=1)
    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    df_between = n - 1

    # Within-candidate: variance within each row
    ss_within = np.sum((values_matrix - row_means[:, np.newaxis]) ** 2)
    df_within = n * (k - 1)

    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0

    # ICC(1,1)
    denom = ms_between + (k - 1) * ms_within
    icc = (ms_between - ms_within) / denom if denom > 0 else 0

    return float(icc), float(ms_between), float(ms_within)


def compute_icc_average(values_matrix):
    """
    Compute ICC(1,k) — reliability of the MEAN of k measurements.

    ICC(1,k) = (MS_between - MS_within) / MS_between
    """
    n, k = values_matrix.shape
    if n < 2 or k < 2:
        return np.nan

    grand_mean = np.mean(values_matrix)
    row_means = np.mean(values_matrix, axis=1)
    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    df_between = n - 1
    ss_within = np.sum((values_matrix - row_means[:, np.newaxis]) ** 2)
    df_within = n * (k - 1)

    ms_between = ss_between / df_between if df_between > 0 else 0
    ms_within = ss_within / df_within if df_within > 0 else 0

    icc_k = (ms_between - ms_within) / ms_between if ms_between > 0 else 0
    return float(icc_k)


def variance_decomposition_3level(candidates):
    """
    Three-level variance decomposition:
      Level 1: Between candidates (signal)
      Level 2: Between WM rollouts within candidate (WM noise)
      Level 3: Between value predictions within WM rollout (value head noise)

    Returns variance components and their proportions.
    """
    n_candidates = len(candidates)
    n_wm = 3
    n_vpred = 5

    # Collect all values in structured form
    all_values = []
    for c in candidates:
        fv = c["full_values"]
        wm_groups = []
        for wm_idx in range(n_wm):
            start = wm_idx * n_vpred
            end = start + n_vpred
            wm_groups.append(fv[start:end])
        all_values.append(wm_groups)

    # Grand mean
    flat = [v for c in all_values for wm in c for v in wm]
    grand_mean = np.mean(flat)

    # Candidate means
    candidate_means = [np.mean([v for wm in c for v in wm]) for c in all_values]

    # WM rollout means (per candidate)
    wm_means = [[np.mean(wm) for wm in c] for c in all_values]

    # Variance components
    # Between-candidate variance
    var_between_candidates = np.var(candidate_means, ddof=0)

    # Between-WM variance (within candidate)
    wm_deviations = []
    for i, c in enumerate(all_values):
        for wm in c:
            wm_deviations.append(np.mean(wm) - candidate_means[i])
    var_between_wm = np.var(wm_deviations, ddof=0)

    # Within-WM variance (value head noise)
    vpred_deviations = []
    for c_idx, c in enumerate(all_values):
        for wm_idx, wm in enumerate(c):
            wm_mean = wm_means[c_idx][wm_idx]
            for v in wm:
                vpred_deviations.append(v - wm_mean)
    var_within_wm = np.var(vpred_deviations, ddof=0)

    total_var = var_between_candidates + var_between_wm + var_within_wm

    return {
        "var_between_candidates": float(var_between_candidates),
        "var_between_wm": float(var_between_wm),
        "var_within_wm": float(var_within_wm),
        "total_var": float(total_var),
        "pct_between_candidates": float(var_between_candidates / total_var * 100) if total_var > 0 else 0,
        "pct_between_wm": float(var_between_wm / total_var * 100) if total_var > 0 else 0,
        "pct_within_wm": float(var_within_wm / total_var * 100) if total_var > 0 else 0,
    }


def analyze_decision_point_stability(dp):
    """Analyze value stability for a single decision point."""
    candidates = dp["candidates"]
    if len(candidates) < 3:
        return None

    n_candidates = len(candidates)

    # Build values matrix: (n_candidates, 15)
    values_matrix = np.array([c["full_values"] for c in candidates])

    # ICC(1,1) — single measurement reliability
    icc_single, ms_between, ms_within = compute_icc_oneway(values_matrix)

    # ICC(1,k) — mean of k=15 measurements reliability
    icc_mean = compute_icc_average(values_matrix)

    # ICC for k=1 (shallow) vs k=3 vs k=5 vs k=15
    icc_by_k = {}
    for k in [1, 3, 5, 15]:
        if k <= values_matrix.shape[1]:
            sub_matrix = values_matrix[:, :k]
            sub_means = np.mean(sub_matrix, axis=1, keepdims=True)
            icc_k_val = compute_icc_average(sub_matrix) if k > 1 else icc_single
            icc_by_k[k] = float(icc_k_val)

    # CV per candidate
    cvs = []
    for c in candidates:
        fv = np.array(c["full_values"])
        mean_v = np.mean(fv)
        std_v = np.std(fv)
        cv = std_v / abs(mean_v) if abs(mean_v) > 1e-8 else np.nan
        cvs.append(cv)

    # Variance decomposition
    var_decomp = variance_decomposition_3level(candidates)

    # Value range across candidates (signal strength)
    candidate_means = [np.mean(c["full_values"]) for c in candidates]
    value_range = max(candidate_means) - min(candidate_means)

    # Mean within-candidate std (noise level)
    mean_within_std = np.mean([np.std(c["full_values"]) for c in candidates])

    # SNR = value_range / mean_within_std
    snr = value_range / mean_within_std if mean_within_std > 1e-8 else np.inf

    return {
        "icc_single": icc_single,
        "icc_mean_15": icc_mean,
        "icc_by_k": icc_by_k,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "cv_mean": float(np.nanmean(cvs)),
        "cv_std": float(np.nanstd(cvs)),
        "var_decomp": var_decomp,
        "value_range": float(value_range),
        "mean_within_std": float(mean_within_std),
        "snr": float(snr),
        "mean_full_value": float(np.mean(candidate_means)),
        "episode_phase": dp.get("episode_phase", "unknown"),
        "timestep": dp.get("timestep", -1),
        "num_candidates": n_candidates,
    }


def run_analysis(input_path, output_dir):
    """Run the full value stability analysis."""
    with open(input_path, "r") as f:
        data = json.load(f)

    config = data["config"]
    print(f"Loaded: {config['task_suite']}, {config['num_candidates']} candidates, "
          f"{config['num_wm_rollouts_full']}WM × {config['num_value_preds_full']}V")

    all_analyses = []
    task_analyses = {}

    for task_data in data["tasks"]:
        task_id = task_data["task_id"]
        task_results = []

        for episode in task_data["episodes"]:
            for dp in episode["decision_points"]:
                result = analyze_decision_point_stability(dp)
                if result is not None:
                    result["task_id"] = task_id
                    result["episode_idx"] = episode["episode_idx"]
                    result["episode_success"] = episode["success"]
                    all_analyses.append(result)
                    task_results.append(result)

        task_analyses[task_id] = task_results

    if not all_analyses:
        print("ERROR: No valid decision points found!")
        return

    print(f"\nAnalyzed {len(all_analyses)} decision points")

    # ==========================================
    # 1. Global ICC statistics
    # ==========================================
    iccs_single = [a["icc_single"] for a in all_analyses if not np.isnan(a["icc_single"])]
    iccs_mean = [a["icc_mean_15"] for a in all_analyses if not np.isnan(a["icc_mean_15"])]
    cvs = [a["cv_mean"] for a in all_analyses if not np.isnan(a["cv_mean"])]
    snrs = [a["snr"] for a in all_analyses if not np.isinf(a["snr"]) and not np.isnan(a["snr"])]

    print(f"\n{'='*60}")
    print(f"VALUE STABILITY — GLOBAL RESULTS")
    print(f"{'='*60}")

    icc_single_mean = float(np.mean(iccs_single))
    icc_mean_mean = float(np.mean(iccs_mean))

    print(f"ICC(1,1) single measurement:  mean={icc_single_mean:.4f}, "
          f"median={np.median(iccs_single):.4f}, std={np.std(iccs_single):.4f}")
    print(f"ICC(1,k=15) mean of 15:       mean={icc_mean_mean:.4f}, "
          f"median={np.median(iccs_mean):.4f}")
    print(f"CV (within-candidate):         mean={np.mean(cvs):.4f}, "
          f"median={np.median(cvs):.4f}")
    print(f"SNR (range/noise):             mean={np.mean(snrs):.2f}, "
          f"median={np.median(snrs):.2f}")

    # ICC interpretation
    if icc_single_mean < 0.5:
        icc_verdict = "POOR — single value prediction cannot reliably distinguish candidates"
    elif icc_single_mean < 0.75:
        icc_verdict = "MODERATE — single prediction has some discriminative power"
    elif icc_single_mean < 0.9:
        icc_verdict = "GOOD — single prediction is fairly reliable"
    else:
        icc_verdict = "EXCELLENT — single prediction is highly reliable"
    print(f"\nICC(1,1) verdict: {icc_verdict}")

    # ==========================================
    # 2. ICC by number of measurements (how many samples needed?)
    # ==========================================
    print(f"\n{'='*60}")
    print(f"ICC BY NUMBER OF MEASUREMENTS (how many samples to stabilize?)")
    print(f"{'='*60}")

    for k in [1, 3, 5, 15]:
        k_iccs = [a["icc_by_k"].get(k, np.nan) for a in all_analyses
                  if not np.isnan(a["icc_by_k"].get(k, np.nan))]
        if k_iccs:
            print(f"  k={k:2d}: ICC={np.mean(k_iccs):.4f} (median={np.median(k_iccs):.4f})")

    # ==========================================
    # 3. Variance decomposition
    # ==========================================
    print(f"\n{'='*60}")
    print(f"VARIANCE DECOMPOSITION (averaged across decision points)")
    print(f"{'='*60}")

    pct_cand = [a["var_decomp"]["pct_between_candidates"] for a in all_analyses]
    pct_wm = [a["var_decomp"]["pct_between_wm"] for a in all_analyses]
    pct_vpred = [a["var_decomp"]["pct_within_wm"] for a in all_analyses]

    print(f"  Between candidates (SIGNAL):     {np.mean(pct_cand):5.1f}%  (median={np.median(pct_cand):.1f}%)")
    print(f"  Between WM rollouts (WM NOISE):  {np.mean(pct_wm):5.1f}%  (median={np.median(pct_wm):.1f}%)")
    print(f"  Within WM (VALUE HEAD NOISE):    {np.mean(pct_vpred):5.1f}%  (median={np.median(pct_vpred):.1f}%)")

    if np.mean(pct_cand) < 50:
        var_verdict = "NOISE DOMINATED — more than half the variance is noise, not signal"
    else:
        var_verdict = "SIGNAL DOMINATED — candidates are distinguishable above noise"
    print(f"\n  Verdict: {var_verdict}")

    # ==========================================
    # 4. Breakdown by episode phase
    # ==========================================
    print(f"\n{'='*60}")
    print(f"BREAKDOWN BY EPISODE PHASE")
    print(f"{'='*60}")

    phase_stats = {}
    for phase in ["early", "mid", "late"]:
        phase_data = [a for a in all_analyses if a["episode_phase"] == phase]
        if phase_data:
            p_iccs = [a["icc_single"] for a in phase_data if not np.isnan(a["icc_single"])]
            p_snrs = [a["snr"] for a in phase_data if not np.isinf(a["snr"]) and not np.isnan(a["snr"])]
            p_pct_cand = [a["var_decomp"]["pct_between_candidates"] for a in phase_data]
            phase_stats[phase] = {
                "count": len(phase_data),
                "icc_single_mean": float(np.mean(p_iccs)) if p_iccs else None,
                "snr_mean": float(np.mean(p_snrs)) if p_snrs else None,
                "pct_signal": float(np.mean(p_pct_cand)),
            }
            print(f"  {phase:5s}: ICC={phase_stats[phase]['icc_single_mean']:.4f}, "
                  f"SNR={phase_stats[phase]['snr_mean']:.2f}, "
                  f"signal%={phase_stats[phase]['pct_signal']:.1f}%, "
                  f"n={phase_stats[phase]['count']}")

    # ==========================================
    # 5. Breakdown by value range
    # ==========================================
    print(f"\n{'='*60}")
    print(f"BREAKDOWN BY VALUE RANGE")
    print(f"{'='*60}")

    mean_values = [a["mean_full_value"] for a in all_analyses]
    v_low = np.percentile(mean_values, 33)
    v_high = np.percentile(mean_values, 67)

    value_range_stats = {}
    for label, lo, hi in [("low", -1, v_low), ("mid", v_low, v_high), ("high", v_high, 2)]:
        group = [a for a in all_analyses if lo <= a["mean_full_value"] < hi]
        if group:
            g_iccs = [a["icc_single"] for a in group if not np.isnan(a["icc_single"])]
            g_pct = [a["var_decomp"]["pct_between_candidates"] for a in group]
            value_range_stats[label] = {
                "count": len(group),
                "icc_single_mean": float(np.mean(g_iccs)) if g_iccs else None,
                "pct_signal": float(np.mean(g_pct)),
            }
            print(f"  {label:4s}: ICC={value_range_stats[label]['icc_single_mean']:.4f}, "
                  f"signal%={value_range_stats[label]['pct_signal']:.1f}%, "
                  f"n={value_range_stats[label]['count']}")

    # ==========================================
    # 6. Per-task breakdown
    # ==========================================
    print(f"\n{'='*60}")
    print(f"PER-TASK BREAKDOWN")
    print(f"{'='*60}")

    per_task_stats = {}
    for task_id, task_results in task_analyses.items():
        if task_results:
            t_iccs = [a["icc_single"] for a in task_results if not np.isnan(a["icc_single"])]
            t_pct = [a["var_decomp"]["pct_between_candidates"] for a in task_results]
            per_task_stats[task_id] = {
                "count": len(task_results),
                "icc_single_mean": float(np.mean(t_iccs)) if t_iccs else None,
                "pct_signal": float(np.mean(t_pct)),
            }
            print(f"  Task {task_id}: ICC={per_task_stats[task_id]['icc_single_mean']:.4f}, "
                  f"signal%={per_task_stats[task_id]['pct_signal']:.1f}%, "
                  f"n={per_task_stats[task_id]['count']}")

    # ==========================================
    # 7. Save results
    # ==========================================
    results = {
        "config": config,
        "experiment": "0-Zero-A Phase 1: Value Stability (from existing data)",
        "num_decision_points": len(all_analyses),
        "global": {
            "icc_single": {
                "mean": float(np.mean(iccs_single)),
                "median": float(np.median(iccs_single)),
                "std": float(np.std(iccs_single)),
                "q25": float(np.percentile(iccs_single, 25)),
                "q75": float(np.percentile(iccs_single, 75)),
            },
            "icc_mean_15": {
                "mean": float(np.mean(iccs_mean)),
                "median": float(np.median(iccs_mean)),
            },
            "icc_by_k": {
                k: float(np.mean([a["icc_by_k"].get(k, np.nan) for a in all_analyses
                                   if not np.isnan(a["icc_by_k"].get(k, np.nan))]))
                for k in [1, 3, 5, 15]
            },
            "cv": {
                "mean": float(np.mean(cvs)),
                "median": float(np.median(cvs)),
                "std": float(np.std(cvs)),
            },
            "snr": {
                "mean": float(np.mean(snrs)),
                "median": float(np.median(snrs)),
            },
            "variance_decomposition": {
                "pct_between_candidates_mean": float(np.mean(pct_cand)),
                "pct_between_candidates_median": float(np.median(pct_cand)),
                "pct_between_wm_mean": float(np.mean(pct_wm)),
                "pct_between_wm_median": float(np.median(pct_wm)),
                "pct_within_wm_mean": float(np.mean(pct_vpred)),
                "pct_within_wm_median": float(np.median(pct_vpred)),
            },
            "icc_verdict": icc_verdict,
            "variance_verdict": var_verdict,
        },
        "by_episode_phase": phase_stats,
        "by_value_range": value_range_stats,
        "by_task": per_task_stats,
        "per_state_icc_distribution": [float(a["icc_single"]) for a in all_analyses],
        "per_state_snr_distribution": [float(a["snr"]) for a in all_analyses
                                        if not np.isinf(a["snr"]) and not np.isnan(a["snr"])],
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "value_stability_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 0-Zero-A: Value Stability Analysis")
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
