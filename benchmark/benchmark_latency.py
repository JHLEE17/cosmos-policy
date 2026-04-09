"""
Cosmos Policy Latency Benchmark - E2E + Component Breakdown

Wraps the proven get_action() path with CUDA event timing.
For component breakdown, times the internal generate_samples_from_batch() by
monkey-patching the model's sampler to record denoising loop time.
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


def get_suite_config(suite):
    configs = {
        "libero": {
            "action_dim": 7, "proprio_dim": 9, "chunk_size": 16,
            "config": "cosmos_predict2_2b_480p_libero__inference_only",
            "ckpt_path": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
            "t5_emb_path": "nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
            "task_label": "pick up the black bowl on the stove and place it on the plate",
        },
        "aloha": {
            "action_dim": 14, "proprio_dim": 14, "chunk_size": 50,
            "config": "cosmos_predict2_2b_480p_aloha_185_demos_4_tasks_mixture_foldshirt15_candiesinbowl45_candyinbag45_eggplantchickenonplate80__inference_only",
            "ckpt_path": "nvidia/Cosmos-Policy-ALOHA-Predict2-2B",
            "t5_emb_path": "nvidia/Cosmos-Policy-ALOHA-Predict2-2B/aloha_t5_embeddings.pkl",
            "task_label": "put the eggplant on the plate",
        },
    }
    return configs[suite]


def build_cfg(suite, sc):
    return SimpleNamespace(
        suite=suite, model_family="cosmos", config=sc["config"], ckpt_path=sc["ckpt_path"],
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True, num_third_person_images=(2 if suite == "robocasa" else 1),
        use_wrist_image=True, num_wrist_images=(2 if suite == "aloha" else 1),
        use_proprio=True, flip_images=(suite in ["libero", "robocasa"]),
        use_jpeg_compression=False, trained_with_image_aug=True,
        use_variance_scale=False, normalize_proprio=True, unnormalize_actions=False,
        chunk_size=sc["chunk_size"], num_open_loop_steps=sc["chunk_size"],
        ar_future_prediction=False, ar_value_prediction=False, ar_qvalue_prediction=False,
        num_denoising_steps_action=5,
        t5_text_embeddings_path=sc["t5_emb_path"], dataset_stats_path="",
    )


def create_obs(suite):
    s = 224
    if suite == "libero":
        return {"wrist_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
                "primary_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
                "proprio": np.random.randn(9).astype(np.float64)}
    elif suite == "aloha":
        return {"left_wrist_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
                "right_wrist_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
                "primary_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
                "proprio": np.random.randn(14).astype(np.float64)}


def instrument_model_for_breakdown(model):
    """Monkey-patch model to capture component-level timing via CUDA events."""
    # Wrap generate_samples_from_batch to time its internal phases
    original_generate = model.generate_samples_from_batch.__func__

    def timed_generate(self, data_batch, **kwargs):
        evt = lambda: torch.cuda.Event(enable_timing=True)

        # Time preprocessing + VAE encode + conditioning setup
        vae_cond_s, vae_cond_e = evt(), evt()
        vae_cond_s.record()

        # Call the preprocessing part (same as original)
        self._normalize_video_databatch_inplace(data_batch)
        self._augment_image_dim_inplace(data_batch)
        is_image_batch = self.is_image_batch(data_batch)
        input_key = self.input_image_key if is_image_batch else self.input_data_key
        n_sample = kwargs.get("n_sample") or data_batch[input_key].shape[0]
        _T, _H, _W = data_batch[input_key].shape[-3:]
        state_shape = [
            self.config.state_ch,
            self.tokenizer.get_latent_num_frames(_T),
            _H // self.tokenizer.spatial_compression_factor,
            _W // self.tokenizer.spatial_compression_factor,
        ]

        return_orig = kwargs.get("return_orig_clean_latent_frames", False)
        if return_orig:
            x0_fn, orig_clean = self.get_x0_fn_from_batch(
                data_batch, kwargs.get("guidance", 0),
                is_negative_prompt=kwargs.get("is_negative_prompt", False),
                return_orig_clean_latent_frames=True)
        else:
            x0_fn = self.get_x0_fn_from_batch(
                data_batch, kwargs.get("guidance", 0),
                is_negative_prompt=kwargs.get("is_negative_prompt", False))

        vae_cond_e.record()

        # Time denoising loop
        from cosmos_policy._src.imaginaire.utils import misc
        denoise_s, denoise_e = evt(), evt()
        denoise_s.record()

        seed = kwargs.get("seed", 1)
        x_sigma_max = kwargs.get("x_sigma_max")
        if x_sigma_max is None:
            x_sigma_max = (
                misc.arch_invariant_rand(
                    (n_sample,) + tuple(state_shape), torch.float32,
                    self.tensor_kwargs["device"], seed)
                * self.sde.sigma_max)

        samples = self.sampler(
            x0_fn, x_sigma_max,
            num_steps=kwargs.get("num_steps", 5),
            sigma_max=self.sde.sigma_max,
            sigma_min=self.sde.sigma_min,
            solver_option=kwargs.get("solver_option", "2ab"))

        denoise_e.record()

        # Store timing events on the model for later retrieval
        self._breakdown_events = {
            "vae_and_conditioning": (vae_cond_s, vae_cond_e),
            "denoising_loop": (denoise_s, denoise_e),
        }

        if return_orig:
            return samples, orig_clean
        return samples

    # Only patch if not already patched
    if not hasattr(model, '_original_generate'):
        model._original_generate = original_generate
        import types
        model.generate_samples_from_batch = types.MethodType(timed_generate, model)
    model._breakdown_events = None


def run_benchmark_for_steps(model, cfg, sc, obs, task_label, dataset_stats,
                             num_steps, num_warmup, num_measure):
    """Run benchmark for a single num_steps value."""
    from cosmos_policy.experiments.robot.cosmos_utils import get_action

    print(f"\n{'='*60}")
    print(f"  Denoising Steps: {num_steps}")
    print(f"{'='*60}")

    cfg.num_denoising_steps_action = num_steps

    # Warmup
    print(f"  Warmup ({num_warmup} iters)...")
    for i in range(num_warmup):
        with torch.inference_mode():
            get_action(cfg, model, dataset_stats, obs, task_label,
                       seed=i, num_denoising_steps_action=num_steps,
                       generate_future_state_and_value_in_parallel=False)

    # Measure
    print(f"  Measuring ({num_measure} iters)...")
    all_timings = []
    for i in range(num_measure):
        torch.cuda.synchronize()
        evt = lambda: torch.cuda.Event(enable_timing=True)

        # E2E timing
        e2e_s, e2e_e = evt(), evt()
        wall_start = time.perf_counter()
        e2e_s.record()

        with torch.inference_mode():
            get_action(cfg, model, dataset_stats, obs, task_label,
                       seed=42, num_denoising_steps_action=num_steps,
                       generate_future_state_and_value_in_parallel=False)

        e2e_e.record()
        torch.cuda.synchronize()
        wall_total = time.perf_counter() - wall_start

        timing = {
            "wall_total": wall_total,
            "gpu_e2e": e2e_s.elapsed_time(e2e_e) / 1000,
        }

        # Extract component breakdown from monkey-patched model
        if model._breakdown_events is not None:
            for comp_name, (cs, ce) in model._breakdown_events.items():
                timing[comp_name] = cs.elapsed_time(ce) / 1000
            timing["other"] = timing["gpu_e2e"] - timing.get("vae_and_conditioning", 0) - timing.get("denoising_loop", 0)

        all_timings.append(timing)

        if (i + 1) % 10 == 0:
            dl = timing.get('denoising_loop', 0) * 1000
            vc = timing.get('vae_and_conditioning', 0) * 1000
            print(f"    [{i+1}/{num_measure}] wall={timing['wall_total']*1000:.1f}ms "
                  f"denoise={dl:.1f}ms vae+cond={vc:.1f}ms")

    # Statistics
    stats = {}
    for key in all_timings[0].keys():
        vals = [t[key] * 1000 for t in all_timings]
        stats[key] = {"mean_ms": float(np.mean(vals)), "std_ms": float(np.std(vals)),
                       "min_ms": float(np.min(vals)), "max_ms": float(np.max(vals)),
                       "median_ms": float(np.median(vals))}

    # Print
    print(f"\n  Results (steps={num_steps}):")
    print(f"  {'─'*58}")
    for key, s in stats.items():
        pct = ""
        if key not in ["wall_total", "gpu_e2e"] and "gpu_e2e" in stats:
            pct = f"  ({s['mean_ms']/stats['gpu_e2e']['mean_ms']*100:5.1f}%)"
        print(f"    {key:24s}: {s['mean_ms']:8.2f} ± {s['std_ms']:5.2f} ms{pct}")
    print(f"  {'─'*58}")

    return {"stats": stats, "raw_timings_ms": [{k: v*1000 for k, v in t.items()} for t in all_timings]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="libero", choices=["libero", "aloha"])
    parser.add_argument("--num-steps", type=int, nargs="+", default=[5])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-measure", type=int, default=30)
    parser.add_argument("--experiment-name", default="benchmark")
    args = parser.parse_args()

    sc = get_suite_config(args.suite)
    cfg = build_cfg(args.suite, sc)

    print(f"{'='*60}")
    print(f"  Cosmos Policy Latency Benchmark")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Suite: {args.suite}, Steps: {args.num_steps}")
    print(f"  Warmup: {args.num_warmup}, Measure: {args.num_measure}")
    print(f"{'='*60}")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model, init_t5_text_embeddings_cache, get_t5_embedding_from_cache)

    print("\nLoading model...")
    t0 = time.perf_counter()
    model, _ = get_model(cfg)
    load_time = time.perf_counter() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Instrument model for component breakdown
    instrument_model_for_breakdown(model)

    init_t5_text_embeddings_cache(sc["t5_emb_path"])
    _ = get_t5_embedding_from_cache(sc["task_label"])  # Pre-cache the embedding
    task_label = sc["task_label"]  # Pass string to get_action (it looks up from cache)

    obs = create_obs(args.suite)
    ad, pd = sc["action_dim"], sc["proprio_dim"]
    dataset_stats = {"actions_min": np.full(ad, -1.0), "actions_max": np.full(ad, 1.0),
                     "proprio_min": np.full(pd, -1.0), "proprio_max": np.full(pd, 1.0)}

    # Metadata
    gp = torch.cuda.get_device_properties(0)
    metadata = {
        "gpu_name": gp.name, "vram_gb": round(gp.total_memory / (1024**3), 1),
        "compute_capability": [gp.major, gp.minor],
        "suite": args.suite, "chunk_size": sc["chunk_size"],
        "action_dim": ad, "proprio_dim": pd,
        "num_warmup": args.num_warmup, "num_measure": args.num_measure,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "model_load_time_s": round(load_time, 1),
    }

    # Run all step configs
    all_results = {}
    for ns in args.num_steps:
        all_results[f"steps_{ns}"] = run_benchmark_for_steps(
            model, cfg, sc, obs, task_label, dataset_stats,
            ns, args.num_warmup, args.num_measure)

    # Save
    output_dir = os.path.join("benchmark", "results", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "latency_results.json")
    with open(path, "w") as f:
        json.dump({"results": all_results, "metadata": metadata}, f, indent=2)
    print(f"\nResults saved to {path}")

    # Final comparison
    print(f"\n{'='*80}")
    print(f"  COMPARISON ACROSS DENOISING STEPS ({args.suite.upper()})")
    print(f"{'='*80}")
    hdr = f"  {'Steps':>5} | {'Wall(ms)':>10} | {'GPU E2E(ms)':>12} | {'Denoise(ms)':>12} | {'VAE+Cond(ms)':>13} | {'Other(ms)':>10} | {'Denoise%':>8}"
    print(hdr)
    print(f"  {'─'*5}-+-{'─'*10}-+-{'─'*12}-+-{'─'*12}-+-{'─'*13}-+-{'─'*10}-+-{'─'*8}")
    for ns in args.num_steps:
        s = all_results[f"steps_{ns}"]["stats"]
        w = s["wall_total"]["mean_ms"]
        e = s["gpu_e2e"]["mean_ms"]
        d = s.get("denoising_loop", {}).get("mean_ms", 0)
        v = s.get("vae_and_conditioning", {}).get("mean_ms", 0)
        o = s.get("other", {}).get("mean_ms", 0)
        dp = d / e * 100 if e > 0 else 0
        print(f"  {ns:5d} | {w:10.1f} | {e:12.1f} | {d:12.1f} | {v:13.1f} | {o:10.1f} | {dp:7.1f}%")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
