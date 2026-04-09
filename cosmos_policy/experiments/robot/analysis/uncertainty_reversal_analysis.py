# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment 0-B Analysis: Uncertainty-Rank Reversal Relationship.

Reads the JSON output from collect_candidate_data.py and tests whether
high ensemble disagreement (value std) predicts rank reversal under full verification.

Tests:
1. Chi-squared test: high vs low uncertainty groups' reversal rates
2. AUC for reversal prediction using different uncertainty signals
3. Comparison of uncertainty metrics: ensemble std vs value margin vs ranking stability

Usage:
    python -m cosmos_policy.experiments.robot.analysis.uncertainty_reversal_analysis \
        --input_path ../../adaptive-ttc-wam/experiments/0A_rank_correlation/candidate_data_libero_spatial_*.json \
        --output_dir ../../adaptive-ttc-wam/experiments/0B_uncertainty_reversal
"""

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats


def compute_rank_reversal(shallow_values, full_values, threshold=2):
    """
    Compute per-candidate rank reversal.

    A candidate experiences "rank reversal" if its rank changes by >= threshold
    between shallow and full verification.

    Returns:
        list[bool]: Whether each candidate experienced rank reversal
        list[float]: Rank change (absolute) for each candidate; fractional when ties exist
    """
    n = len(shallow_values)
    # Rank: 0 = lowest value, n-1 = highest
    shallow_ranks = stats.rankdata(shallow_values) - 1
    full_ranks = stats.rankdata(full_values) - 1

    # Keep fractional ranks to handle ties correctly (avoid int() truncation bias)
    rank_changes = [abs(float(shallow_ranks[i]) - float(full_ranks[i])) for i in range(n)]
    reversals = [rc >= threshold for rc in rank_changes]

    return reversals, rank_changes


def compute_uncertainty_signals(candidates):
    """
    Compute multiple uncertainty signals for each candidate.

    Signals are categorized as:
      DEPLOYABLE  - computed from shallow values only; available before full verification
                    and usable in the adaptive algorithm at decision time.
      DIAGNOSTIC  - require full verification values (all 15 rollouts); NOT available
                    at decision time and must not be used for online allocation.

    Returns dict mapping signal name -> list of values (one per candidate).
    """
    n = len(candidates)

    # --- DIAGNOSTIC signals (require full verification data) ---

    # Signal 1: Ensemble disagreement (std of all individual value predictions)
    # DIAGNOSTIC: uses full_value_std, which requires all verification rollouts.
    ensemble_stds = [c["full_value_std"] for c in candidates]

    # Signal 3: Intra-WM variance (variance across WM rollouts)
    # DIAGNOSTIC: uses full_values, which requires all verification rollouts.
    intra_wm_stds = []
    for c in candidates:
        full_values = c["full_values"]
        num_wm = len(full_values) // max(1, len(full_values) // 3) if len(full_values) >= 3 else 1
        # Group values by WM rollout
        vals_per_wm = len(full_values) // num_wm if num_wm > 0 else len(full_values)
        wm_means = []
        for wm_idx in range(num_wm):
            start = wm_idx * vals_per_wm
            end = start + vals_per_wm
            wm_means.append(np.mean(full_values[start:end]))
        intra_wm_stds.append(float(np.std(wm_means)) if len(wm_means) > 1 else 0.0)

    # --- DEPLOYABLE signals (shallow values only) ---

    # Signal 2: Value margin (distance from best candidate)
    # DEPLOYABLE: uses only shallow_value.
    shallow_values = [c["shallow_value"] for c in candidates]
    best_shallow = max(shallow_values)
    value_margins = [best_shallow - v for v in shallow_values]

    # Signal 4: Shallow value gap (difference between a candidate's shallow value
    # and the shallow value of the next-best candidate).
    # DEPLOYABLE: uses only shallow_value; directly available at decision time.
    sorted_shallow = sorted(shallow_values, reverse=True)
    shallow_value_gaps = []
    for sv in shallow_values:
        idx = sorted_shallow.index(sv)
        if idx == 0:
            # Best candidate: gap to second-best
            gap = sorted_shallow[0] - sorted_shallow[1] if len(sorted_shallow) > 1 else 0.0
        else:
            # Other candidates: gap to the candidate ranked just above them
            gap = sorted_shallow[idx - 1] - sv
        shallow_value_gaps.append(gap)

    return {
        # DIAGNOSTIC signals (not deployable)
        "ensemble_std": ensemble_stds,
        "intra_wm_std": intra_wm_stds,
        # DEPLOYABLE signals
        "value_margin": value_margins,
        "shallow_value_gap": shallow_value_gaps,
    }


def chi_squared_test(high_reversals, low_reversals):
    """Run chi-squared test comparing reversal rates between high and low uncertainty groups."""
    high_total = len(high_reversals)
    low_total = len(low_reversals)
    high_reversal_count = sum(high_reversals)
    low_reversal_count = sum(low_reversals)

    # Contingency table
    table = np.array([
        [high_reversal_count, high_total - high_reversal_count],
        [low_reversal_count, low_total - low_reversal_count],
    ])

    if table.min() == 0 and (high_total + low_total) < 40:
        # Use Fisher's exact test for small samples
        odds_ratio, p_value = stats.fisher_exact(table)
        test_name = "fisher_exact"
        stat = odds_ratio
    else:
        stat, p_value, dof, expected = stats.chi2_contingency(table, correction=True)
        test_name = "chi_squared"

    # Cramer's V effect size
    n = high_total + low_total
    cramers_v = np.sqrt(stat / n) if test_name == "chi_squared" and n > 0 else np.nan

    return {
        "test": test_name,
        "statistic": float(stat),
        "p_value": float(p_value),
        "cramers_v": float(cramers_v) if not np.isnan(cramers_v) else None,
        "high_uncertainty": {
            "total": high_total,
            "reversals": high_reversal_count,
            "rate": high_reversal_count / high_total if high_total > 0 else 0,
        },
        "low_uncertainty": {
            "total": low_total,
            "reversals": low_reversal_count,
            "rate": low_reversal_count / low_total if low_total > 0 else 0,
        },
    }


def compute_auc_for_reversal_prediction(uncertainty_values, is_reversal):
    """Compute AUC-ROC for using uncertainty to predict rank reversal.

    Uses the Mann-Whitney U statistic: AUC = U / (n_pos * n_neg).
    This handles tied uncertainty values correctly without sorting-order bias.
    """
    if sum(is_reversal) == 0 or sum(is_reversal) == len(is_reversal):
        return np.nan  # Can't compute AUC with only one class

    pos_scores = [u for u, r in zip(uncertainty_values, is_reversal) if r]
    neg_scores = [u for u, r in zip(uncertainty_values, is_reversal) if not r]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    # U statistic: count pairs where positive score > negative score;
    # ties contribute 0.5 each (proper handling via mannwhitneyu).
    result = stats.mannwhitneyu(pos_scores, neg_scores, alternative="greater")
    auc = result.statistic / (n_pos * n_neg)
    return float(auc)


def run_analysis(input_path, output_dir):
    """Run the full uncertainty-reversal analysis."""
    with open(input_path, "r") as f:
        data = json.load(f)

    config = data["config"]
    print(f"Loaded: {config['task_suite']}, {config['num_candidates']} candidates")

    # Collect all per-candidate data across decision points
    all_candidates_flat = []  # Each entry: {uncertainty signals, reversal info, metadata}

    for task_data in data["tasks"]:
        for episode in task_data["episodes"]:
            for dp in episode["decision_points"]:
                candidates = dp["candidates"]
                if len(candidates) < 3:
                    continue

                shallow_values = [c["shallow_value"] for c in candidates]
                # Exclude shallow_value from the full-verification mean to avoid target leakage:
                # full_value_mean typically includes shallow_value as its first sample, which
                # would artificially inflate rank correlation between shallow and full ranks.
                full_values = [
                    float(np.mean(c["full_values"][1:])) if len(c.get("full_values", [])) > 1
                    else c["full_value_mean"]
                    for c in candidates
                ]

                # Compute reversals
                reversals, rank_changes = compute_rank_reversal(shallow_values, full_values, threshold=2)

                # Compute uncertainty signals
                uncertainty_signals = compute_uncertainty_signals(candidates)

                for i, c in enumerate(candidates):
                    all_candidates_flat.append({
                        "ensemble_std": uncertainty_signals["ensemble_std"][i],
                        "value_margin": uncertainty_signals["value_margin"][i],
                        "intra_wm_std": uncertainty_signals["intra_wm_std"][i],
                        "shallow_value_gap": uncertainty_signals["shallow_value_gap"][i],
                        "is_reversal": reversals[i],
                        "rank_change": rank_changes[i],
                        "shallow_value": c["shallow_value"],
                        "full_value_mean": c["full_value_mean"],
                        "task_id": task_data["task_id"],
                        "episode_phase": dp.get("episode_phase", "unknown"),
                    })

    if not all_candidates_flat:
        print("ERROR: No valid candidates found!")
        return

    n_total = len(all_candidates_flat)
    n_reversals = sum(1 for c in all_candidates_flat if c["is_reversal"])
    print(f"\nTotal candidates: {n_total}")
    print(f"Reversals (rank change >= 2): {n_reversals} ({n_reversals/n_total:.1%})")

    # ==========================================
    # 1. Chi-squared test for each uncertainty signal
    # ==========================================
    # NOTE: Candidates within a decision point are not independent (they share the same
    # world-model rollouts and context), so the iid assumption of chi-squared is violated.
    # Treat p-values as approximate/liberal; use effect sizes (Cramer's V) as primary evidence.
    chi_sq_results = {}
    for signal_name in ["ensemble_std", "value_margin", "intra_wm_std", "shallow_value_gap"]:
        values = [c[signal_name] for c in all_candidates_flat]
        median_val = np.median(values)

        high_group = [c["is_reversal"] for c in all_candidates_flat if c[signal_name] >= median_val]
        low_group = [c["is_reversal"] for c in all_candidates_flat if c[signal_name] < median_val]

        result = chi_squared_test(high_group, low_group)
        chi_sq_results[signal_name] = result

        print(f"\n{'='*60}")
        print(f"CHI-SQUARED TEST: {signal_name}")
        print(f"{'='*60}")
        print(f"  High uncertainty reversal rate: {result['high_uncertainty']['rate']:.1%} "
              f"({result['high_uncertainty']['reversals']}/{result['high_uncertainty']['total']})")
        print(f"  Low uncertainty reversal rate:  {result['low_uncertainty']['rate']:.1%} "
              f"({result['low_uncertainty']['reversals']}/{result['low_uncertainty']['total']})")
        print(f"  Test: {result['test']}, stat={result['statistic']:.3f}, p={result['p_value']:.4f}")
        if result['cramers_v'] is not None:
            print(f"  Cramer's V: {result['cramers_v']:.3f}")

        sig = "YES" if result['p_value'] < 0.05 else "NO"
        print(f"  Significant (p < 0.05)? {sig} (approximate; iid assumption violated within a decision point)")

    # ==========================================
    # 2. AUC for reversal prediction
    # ==========================================
    # Signals are separated into deployable (available at decision time) and
    # diagnostic (require full verification; for post-hoc analysis only).
    DIAGNOSTIC_SIGNALS = {"ensemble_std", "intra_wm_std"}
    DEPLOYABLE_SIGNALS = {"value_margin", "shallow_value_gap"}

    auc_results = {}
    auc_deployable = {}
    auc_diagnostic = {}

    print(f"\n{'='*60}")
    print(f"AUC-ROC FOR REVERSAL PREDICTION")
    print(f"{'='*60}")

    is_reversal = [c["is_reversal"] for c in all_candidates_flat]
    for signal_name in ["ensemble_std", "value_margin", "intra_wm_std", "shallow_value_gap"]:
        values = [c[signal_name] for c in all_candidates_flat]
        auc = compute_auc_for_reversal_prediction(values, is_reversal)
        auc_results[signal_name] = float(auc)
        label = "DIAGNOSTIC" if signal_name in DIAGNOSTIC_SIGNALS else "DEPLOYABLE"
        print(f"  {signal_name:22s} [{label:10s}]: AUC = {auc:.3f}")
        if signal_name in DIAGNOSTIC_SIGNALS:
            auc_diagnostic[signal_name] = float(auc)
        else:
            auc_deployable[signal_name] = float(auc)

    # ==========================================
    # 3. Rank change distribution
    # ==========================================
    rank_changes = [c["rank_change"] for c in all_candidates_flat]
    rank_change_dist = {
        "mean": float(np.mean(rank_changes)),
        "median": float(np.median(rank_changes)),
        "std": float(np.std(rank_changes)),
        "histogram": {f"{v:.1f}": int(rank_changes.count(v)) for v in sorted(set(rank_changes))},
    }

    print(f"\n{'='*60}")
    print(f"RANK CHANGE DISTRIBUTION")
    print(f"{'='*60}")
    print(f"  Mean rank change: {rank_change_dist['mean']:.2f}")
    print(f"  Median: {rank_change_dist['median']:.1f}")
    for rc, count in sorted(rank_change_dist["histogram"].items(), key=lambda x: float(x[0])):
        bar = "#" * max(1, int(count / n_total * 200))
        print(f"  |delta|={rc}: {count:4d} ({count/n_total:.1%}) {bar}")

    # ==========================================
    # 4. Summary and conclusion
    # ==========================================
    # Best uncertainty signal = highest AUC
    best_signal = max(auc_results, key=auc_results.get)
    best_auc = auc_results[best_signal]

    # Is the relationship significant?
    best_chi_sq = chi_sq_results[best_signal]
    is_significant = best_chi_sq["p_value"] < 0.05

    print(f"\n{'='*60}")
    print(f"CONCLUSION")
    print(f"{'='*60}")
    print(f"  Best uncertainty signal: {best_signal} (AUC={best_auc:.3f})")
    print(f"  Significant relationship: {'YES' if is_significant else 'NO'} (p={best_chi_sq['p_value']:.4f})")

    if is_significant and best_auc > 0.55:
        conclusion = "CONFIRMED"
        msg = "High uncertainty significantly predicts rank reversal -> uncertainty-aware allocation is justified"
    elif is_significant:
        conclusion = "WEAK"
        msg = "Statistically significant but low effect size -> proceed with caution"
    else:
        conclusion = "INCONCLUSIVE"
        msg = "No significant relationship found -> simple top-k pruning may suffice"

    print(f"  Verdict: {conclusion}")
    print(f"  {msg}")

    # ==========================================
    # 5. Save results
    # ==========================================
    results = {
        "config": config,
        "total_candidates": n_total,
        "total_reversals": n_reversals,
        "reversal_rate": n_reversals / n_total,
        "chi_squared_tests": chi_sq_results,
        "auc_reversal_prediction": auc_results,
        "auc_deployable": auc_deployable,
        "auc_diagnostic": auc_diagnostic,
        "best_uncertainty_signal": best_signal,
        "rank_change_distribution": rank_change_dist,
        "conclusion": conclusion,
        "conclusion_message": msg,
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "uncertainty_reversal_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Exp 0-B: Uncertainty-Rank Reversal Analysis")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to candidate_data JSON from collect_candidate_data.py")
    parser.add_argument("--output_dir", type=str,
                        default="../../adaptive-ttc-wam/experiments/0B_uncertainty_reversal",
                        help="Output directory for results")
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
