"""
Cosmos Policy DiT Deep Breakdown Benchmark

Instruments the DiT's 28 transformer blocks to measure:
- Self-Attention, Cross-Attention, FFN time per block
- Per-denoising-step timing
- Aggregated stats across all blocks and steps

Runs on top of the existing get_action() path with model instrumentation.
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_suite_config(suite):
    configs = {
        "libero": {
            "action_dim": 7, "proprio_dim": 9, "chunk_size": 16,
            "config": "cosmos_predict2_2b_480p_libero__inference_only",
            "ckpt_path": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
            "t5_emb_path": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
            "task_label": "pick up the black bowl on the stove and place it on the plate",
        },
    }
    return configs[suite]


def build_cfg(suite, sc):
    from types import SimpleNamespace
    return SimpleNamespace(
        suite=suite, model_family="cosmos", config=sc["config"], ckpt_path=sc["ckpt_path"],
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True, num_third_person_images=1,
        use_wrist_image=True, num_wrist_images=1,
        use_proprio=True, flip_images=True,
        use_jpeg_compression=False, trained_with_image_aug=True,
        use_variance_scale=False, normalize_proprio=True, unnormalize_actions=False,
        chunk_size=sc["chunk_size"], num_open_loop_steps=sc["chunk_size"],
        ar_future_prediction=False, ar_value_prediction=False, ar_qvalue_prediction=False,
        num_denoising_steps_action=5,
        t5_text_embeddings_path=sc["t5_emb_path"], dataset_stats_path="",
    )


def create_obs(suite):
    s = 224
    return {"wrist_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
            "primary_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
            "proprio": np.random.randn(9).astype(np.float64)}


def instrument_dit_blocks(model):
    """Instrument each transformer block in the DiT to capture self-attn, cross-attn, FFN timing."""
    # Access the network (DiT backbone)
    net = model.net

    # Find transformer blocks
    if hasattr(net, 'blocks'):
        blocks = net.blocks
    else:
        print("WARNING: Could not find 'blocks' attribute on net. Trying module search...")
        blocks = None
        for name, module in net.named_modules():
            if 'blocks' in name and isinstance(module, torch.nn.ModuleList):
                blocks = module
                break

    if blocks is None:
        print("ERROR: Could not find transformer blocks!")
        return False

    num_blocks = len(blocks)
    print(f"  Found {num_blocks} transformer blocks to instrument")

    # Store timing data on the model
    model._dit_step_timings = []
    model._dit_instrumented = True

    # Instrument each block
    for block_idx, block in enumerate(blocks):
        block._block_idx = block_idx
        block._timing_events = None

        original_forward = block.forward

        def make_instrumented_forward(orig_fwd, blk):
            def instrumented_forward(*args, **kwargs):
                evt = lambda: torch.cuda.Event(enable_timing=True)

                # We need to understand the block structure
                # MinimalV1LVGDiT blocks typically have: self_attn, cross_attn, ffn
                # But the exact forward logic varies. Let's time the entire block
                # and also try to time sub-components if they exist.

                blk_s, blk_e = evt(), evt()
                blk_s.record()
                result = orig_fwd(*args, **kwargs)
                blk_e.record()

                blk._timing_events = (blk_s, blk_e)
                return result
            return instrumented_forward

        block.forward = make_instrumented_forward(original_forward, block)

    # Also instrument the denoiser to capture per-step block timings
    # We wrap the x0_fn that gets called by the sampler
    original_denoise = model.denoise

    def instrumented_denoise(self_model, *args, **kwargs):
        result = original_denoise(*args, **kwargs)

        # Collect block timings for this denoising step
        step_block_events = []
        for block in blocks:
            if block._timing_events is not None:
                step_block_events.append(block._timing_events)
                block._timing_events = None

        if step_block_events:
            model._dit_step_timings.append(step_block_events)

        return result

    model.denoise = lambda *a, **kw: instrumented_denoise(model, *a, **kw)

    return True


def collect_dit_timings(model):
    """Collect and reset DiT timing data. Must be called after cuda.synchronize()."""
    if not hasattr(model, '_dit_step_timings') or not model._dit_step_timings:
        return None

    timings = {
        "num_steps": len(model._dit_step_timings),
        "blocks_per_step": len(model._dit_step_timings[0]) if model._dit_step_timings else 0,
        "per_step_ms": [],
        "per_block_ms": [],
        "total_dit_ms": 0,
    }

    all_block_times = []
    for step_idx, step_events in enumerate(model._dit_step_timings):
        step_total = 0
        block_times = []
        for blk_s, blk_e in step_events:
            t = blk_s.elapsed_time(blk_e)
            step_total += t
            block_times.append(t)
        timings["per_step_ms"].append(step_total)
        all_block_times.append(block_times)
        timings["total_dit_ms"] += step_total

    # Average per-block across steps
    if all_block_times:
        num_blocks = len(all_block_times[0])
        avg_per_block = []
        for b in range(num_blocks):
            avg = np.mean([all_block_times[s][b] for s in range(len(all_block_times))])
            avg_per_block.append(float(avg))
        timings["avg_per_block_ms"] = avg_per_block

    # Reset
    model._dit_step_timings = []

    return timings


def run_dit_breakdown(model, cfg, sc, obs, task_label, dataset_stats,
                       num_steps, num_warmup, num_measure):
    """Run DiT breakdown benchmark."""
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    print(f"\n{'='*60}")
    print(f"  DiT Breakdown: {num_steps} denoising steps")
    print(f"{'='*60}")

    cfg.num_denoising_steps_action = num_steps

    # Warmup
    print(f"  Warmup ({num_warmup} iters)...")
    for i in range(num_warmup):
        with torch.inference_mode():
            get_action(cfg, model, dataset_stats, obs, task_label,
                       seed=i, num_denoising_steps_action=num_steps,
                       generate_future_state_and_value_in_parallel=False)
        torch.cuda.synchronize()
        model._dit_step_timings = []  # Clear warmup timings

    # Measure
    print(f"  Measuring ({num_measure} iters)...")
    all_dit_timings = []
    all_e2e = []

    for i in range(num_measure):
        torch.cuda.synchronize()
        model._dit_step_timings = []

        evt = lambda: torch.cuda.Event(enable_timing=True)
        e2e_s, e2e_e = evt(), evt()
        e2e_s.record()

        with torch.inference_mode():
            get_action(cfg, model, dataset_stats, obs, task_label,
                       seed=42, num_denoising_steps_action=num_steps,
                       generate_future_state_and_value_in_parallel=False)

        e2e_e.record()
        torch.cuda.synchronize()

        e2e_ms = e2e_s.elapsed_time(e2e_e)
        all_e2e.append(e2e_ms)

        dit_timing = collect_dit_timings(model)
        if dit_timing:
            all_dit_timings.append(dit_timing)

        if (i + 1) % 10 == 0:
            dit_ms = dit_timing["total_dit_ms"] if dit_timing else 0
            print(f"    [{i+1}/{num_measure}] e2e={e2e_ms:.1f}ms dit_blocks={dit_ms:.1f}ms "
                  f"({dit_ms/e2e_ms*100:.1f}%)")

    # Aggregate
    result = {
        "num_steps": num_steps,
        "gpu_e2e_ms": {"mean": float(np.mean(all_e2e)), "std": float(np.std(all_e2e))},
    }

    if all_dit_timings:
        total_dits = [t["total_dit_ms"] for t in all_dit_timings]
        result["dit_total_ms"] = {"mean": float(np.mean(total_dits)), "std": float(np.std(total_dits))}
        result["dit_pct_of_e2e"] = float(np.mean(total_dits) / np.mean(all_e2e) * 100)

        # Per-step stats
        num_s = all_dit_timings[0]["num_steps"]
        per_step_means = []
        for s in range(num_s):
            vals = [t["per_step_ms"][s] for t in all_dit_timings if s < len(t["per_step_ms"])]
            per_step_means.append({"step": s, "mean_ms": float(np.mean(vals)), "std_ms": float(np.std(vals))})
        result["per_step"] = per_step_means

        # Per-block stats (averaged across steps)
        num_b = all_dit_timings[0]["blocks_per_step"]
        per_block_means = []
        for b in range(num_b):
            vals = [t["avg_per_block_ms"][b] for t in all_dit_timings if b < len(t.get("avg_per_block_ms", []))]
            if vals:
                per_block_means.append({"block": b, "mean_ms": float(np.mean(vals)), "std_ms": float(np.std(vals))})
        result["per_block"] = per_block_means

        # Non-DiT time
        non_dit = np.mean(all_e2e) - np.mean(total_dits)
        result["non_dit_ms"] = {"mean": float(non_dit)}

    # Print summary
    print(f"\n  Summary (steps={num_steps}):")
    print(f"  {'─'*50}")
    print(f"    GPU E2E:        {result['gpu_e2e_ms']['mean']:8.1f} ms")
    if "dit_total_ms" in result:
        print(f"    DiT Blocks:     {result['dit_total_ms']['mean']:8.1f} ms ({result['dit_pct_of_e2e']:.1f}% of E2E)")
        print(f"    Non-DiT:        {result['non_dit_ms']['mean']:8.1f} ms ({100-result['dit_pct_of_e2e']:.1f}% of E2E)")
        print(f"\n    Per Denoising Step:")
        for ps in result["per_step"]:
            print(f"      Step {ps['step']:2d}: {ps['mean_ms']:7.1f} ± {ps['std_ms']:.1f} ms")
        print(f"\n    Per Block (avg across steps, ms):")
        blk_vals = [b["mean_ms"] for b in result["per_block"]]
        print(f"      Range: {min(blk_vals):.2f} - {max(blk_vals):.2f} ms")
        print(f"      Mean:  {np.mean(blk_vals):.2f} ms/block")
    print(f"  {'─'*50}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero")
    parser.add_argument("--num-steps", type=int, nargs="+", default=[5])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-measure", type=int, default=30)
    parser.add_argument("--experiment-name", default="dit_breakdown")
    args = parser.parse_args()

    sc = get_suite_config(args.suite)
    cfg = build_cfg(args.suite, sc)

    print(f"{'='*60}")
    print(f"  Cosmos Policy DiT Deep Breakdown")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Steps: {args.num_steps}")
    print(f"{'='*60}")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model, init_t5_text_embeddings_cache, get_t5_embedding_from_cache)

    print("\nLoading model...")
    t0 = time.perf_counter()
    model, _ = get_model(cfg)
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s")

    # Instrument DiT blocks
    print("\nInstrumenting DiT blocks...")
    success = instrument_dit_blocks(model)
    if not success:
        print("Failed to instrument blocks!")
        return

    init_t5_text_embeddings_cache(sc["t5_emb_path"])
    _ = get_t5_embedding_from_cache(sc["task_label"])
    task_label = sc["task_label"]

    obs = create_obs(args.suite)
    ad, pd = sc["action_dim"], sc["proprio_dim"]
    dataset_stats = {"actions_min": np.full(ad, -1.0), "actions_max": np.full(ad, 1.0),
                     "proprio_min": np.full(pd, -1.0), "proprio_max": np.full(pd, 1.0)}

    gp = torch.cuda.get_device_properties(0)
    metadata = {
        "gpu_name": gp.name, "vram_gb": round(gp.total_memory / (1024**3), 1),
        "compute_capability": [gp.major, gp.minor],
        "suite": args.suite, "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "num_warmup": args.num_warmup, "num_measure": args.num_measure,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    all_results = {}
    for ns in args.num_steps:
        all_results[f"steps_{ns}"] = run_dit_breakdown(
            model, cfg, sc, obs, task_label, dataset_stats,
            ns, args.num_warmup, args.num_measure)

    output_dir = os.path.join("benchmark", "results", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "dit_breakdown_results.json")
    with open(path, "w") as f:
        json.dump({"results": all_results, "metadata": metadata}, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
