"""
Cosmos Policy Benchmark Visualization

Generates charts from benchmark results JSON files.

Usage:
    python benchmark/visualize_results.py \
        --pipeline-results benchmark/results/libero_breakdown/latency_results.json \
        --dit-results benchmark/results/dit_subcomponents/dit_subcomponent_results.json \
        --output-dir benchmark/results/figures
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


COLORS = {
    "vae_and_conditioning": "#4C72B0",
    "denoising_loop": "#C44E52",
    "other": "#8172B2",
    "self_attn": "#E24A33",
    "cross_attn": "#348ABD",
    "ffn": "#988ED5",
    "adaln": "#FBC15E",
    "non_dit": "#8EBA42",
}


def plot_pipeline_breakdown(data, output_dir):
    """Plot pipeline-level component breakdown across step counts."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    steps_keys = sorted(data["results"].keys(), key=lambda x: int(x.split("_")[1]))
    step_nums = [int(k.split("_")[1]) for k in steps_keys]

    # ── Panel 1: Stacked bar chart ──
    ax = axes[0]
    components = ["vae_and_conditioning", "denoising_loop", "other"]
    labels = ["VAE + Conditioning", "Denoising Loop", "Other"]
    x = np.arange(len(step_nums))
    width = 0.5

    bottoms = np.zeros(len(step_nums))
    for comp, label in zip(components, labels):
        vals = [data["results"][k]["stats"][comp]["mean_ms"] for k in steps_keys]
        bars = ax.bar(x, vals, width, bottom=bottoms, label=label, color=COLORS[comp])
        # Add value labels
        for j, v in enumerate(vals):
            if v > 15:
                ax.text(x[j], bottoms[j] + v/2, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottoms += vals

    # Add E2E total on top
    for j, k in enumerate(steps_keys):
        e2e = data["results"][k]["stats"]["gpu_e2e"]["mean_ms"]
        ax.text(x[j], bottoms[j] + 8, f"{e2e:.0f}ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} steps" for s in step_nums])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Pipeline Component Breakdown")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: Percentage breakdown ──
    ax = axes[1]
    for i, (k, ns) in enumerate(zip(steps_keys, step_nums)):
        e2e = data["results"][k]["stats"]["gpu_e2e"]["mean_ms"]
        vals = [data["results"][k]["stats"][c]["mean_ms"] / e2e * 100 for c in components]
        bottoms_pct = 0
        for v, comp, label in zip(vals, components, labels):
            ax.barh(i, v, left=bottoms_pct, color=COLORS[comp], edgecolor="white", linewidth=0.5)
            if v > 5:
                ax.text(bottoms_pct + v/2, i, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
            bottoms_pct += v

    ax.set_yticks(range(len(step_nums)))
    ax.set_yticklabels([f"{s} steps" for s in step_nums])
    ax.set_xlabel("Percentage of GPU E2E (%)")
    ax.set_title("Component Proportion")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)

    # ── Panel 3: Per-step denoising cost ──
    ax = axes[2]
    for k, ns in zip(steps_keys, step_nums):
        denoise = data["results"][k]["stats"]["denoising_loop"]["mean_ms"]
        per_step = denoise / ns
        ax.bar(str(ns), per_step, color=COLORS["denoising_loop"], alpha=0.8)
        ax.text(str(ns), per_step + 1, f"{per_step:.1f}ms", ha="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Denoising Steps")
    ax.set_ylabel("Time per Step (ms)")
    ax.set_title("Denoising Cost per Step")
    ax.grid(axis="y", alpha=0.3)

    gpu_name = data.get("metadata", {}).get("gpu_name", "Unknown GPU")
    fig.suptitle(f"Cosmos Policy Pipeline Breakdown — LIBERO — {gpu_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "pipeline_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_dit_subcomponent(data, output_dir):
    """Plot DiT sub-component breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    steps_keys = sorted(data["results"].keys(), key=lambda x: int(x.split("_")[1]))
    step_nums = [int(k.split("_")[1]) for k in steps_keys]
    comp_names = ["self_attn", "cross_attn", "ffn", "adaln"]
    comp_labels = ["Self-Attention", "Cross-Attention", "FFN/MLP", "AdaLN"]

    # ── Panel 1: Stacked bar for DiT internals ──
    ax = axes[0]
    x = np.arange(len(step_nums))
    width = 0.5
    bottoms = np.zeros(len(step_nums))

    for comp, label in zip(comp_names, comp_labels):
        vals = [data["results"][k]["components"][comp]["mean_ms"] for k in steps_keys]
        ax.bar(x, vals, width, bottom=bottoms, label=label, color=COLORS[comp])
        for j, v in enumerate(vals):
            if v > 10:
                ax.text(x[j], bottoms[j] + v/2, f"{v:.0f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        bottoms += vals

    # Add non-DiT
    non_dit_vals = [data["results"][k]["non_dit_ms"]["mean"] for k in steps_keys]
    ax.bar(x, non_dit_vals, width, bottom=bottoms, label="Non-DiT", color=COLORS["non_dit"], alpha=0.6)

    for j, k in enumerate(steps_keys):
        total = data["results"][k]["gpu_e2e_ms"]["mean"]
        ax.text(x[j], bottoms[j] + non_dit_vals[j] + 8, f"{total:.0f}ms",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s} steps" for s in step_nums])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("DiT Sub-Component Breakdown")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ── Panel 2: Pie chart for 5-step ──
    ax = axes[1]
    key_5 = "steps_5" if "steps_5" in data["results"] else steps_keys[1]
    r = data["results"][key_5]
    sizes = [r["components"][c]["mean_ms"] for c in comp_names]
    sizes.append(r["non_dit_ms"]["mean"])
    pie_labels = comp_labels + ["Non-DiT"]
    pie_colors = [COLORS[c] for c in comp_names] + [COLORS["non_dit"]]

    wedges, texts, autotexts = ax.pie(sizes, labels=pie_labels, colors=pie_colors,
                                       autopct="%1.1f%%", startangle=90, textprops={"fontsize": 9})
    for t in autotexts:
        t.set_fontweight("bold")
    ns_label = key_5.split("_")[1]
    ax.set_title(f"Proportion ({ns_label} steps)")

    # ── Panel 3: DiT component % across step counts ──
    ax = axes[2]
    x = np.arange(len(comp_names))
    width = 0.25
    offsets = np.arange(len(step_nums)) - (len(step_nums)-1)/2
    offsets = offsets * width

    for i, (k, ns) in enumerate(zip(steps_keys, step_nums)):
        dit_total = data["results"][k]["dit_block_sum_ms"]["mean"]
        vals = [data["results"][k]["components"][c]["mean_ms"] / dit_total * 100 for c in comp_names]
        bars = ax.bar(x + offsets[i], vals, width, label=f"{ns} steps", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, fontsize=9)
    ax.set_ylabel("% of DiT Block Time")
    ax.set_title("Component Share Within DiT")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    gpu_name = data.get("metadata", {}).get("gpu_name", "Unknown GPU")
    fig.suptitle(f"Cosmos Policy DiT Deep Breakdown — LIBERO — {gpu_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "dit_subcomponent_breakdown.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-results", type=str, default=None)
    parser.add_argument("--dit-results", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="benchmark/results/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Generating visualizations...")

    if args.pipeline_results and os.path.exists(args.pipeline_results):
        with open(args.pipeline_results) as f:
            pipeline_data = json.load(f)
        plot_pipeline_breakdown(pipeline_data, args.output_dir)
    else:
        print(f"  Pipeline results not found: {args.pipeline_results}")

    if args.dit_results and os.path.exists(args.dit_results):
        with open(args.dit_results) as f:
            dit_data = json.load(f)
        plot_dit_subcomponent(dit_data, args.output_dir)
    else:
        print(f"  DiT results not found: {args.dit_results}")

    print("Done!")


if __name__ == "__main__":
    main()
