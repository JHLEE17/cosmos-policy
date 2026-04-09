"""
Cosmos Policy DiT Sub-Component Breakdown

Instruments Block.forward() to separately measure:
  1. AdaLN Modulation (scale/shift/gate computation)
  2. Self-Attention (LayerNorm + self_attn + residual)
  3. Cross-Attention (LayerNorm + cross_attn + residual)
  4. FFN/MLP (LayerNorm + mlp + residual)
"""

import argparse
import json
import os
import sys
import time
import types

import numpy as np
import torch
from einops import rearrange

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def build_cfg(suite="libero"):
    from types import SimpleNamespace
    return SimpleNamespace(
        suite=suite, model_family="cosmos",
        config="cosmos_predict2_2b_480p_libero__inference_only",
        ckpt_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B",
        config_file="cosmos_policy/config/config.py",
        use_third_person_image=True, num_third_person_images=1,
        use_wrist_image=True, num_wrist_images=1,
        use_proprio=True, flip_images=True,
        use_jpeg_compression=False, trained_with_image_aug=True,
        use_variance_scale=False, normalize_proprio=True, unnormalize_actions=False,
        chunk_size=16, num_open_loop_steps=16,
        ar_future_prediction=False, ar_value_prediction=False, ar_qvalue_prediction=False,
        num_denoising_steps_action=5,
        t5_text_embeddings_path="nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl",
        dataset_stats_path="",
    )


def create_obs():
    s = 224
    return {"wrist_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
            "primary_image": np.random.randint(0, 255, (s, s, 3), dtype=np.uint8),
            "proprio": np.random.randn(9).astype(np.float64)}


def instrument_block_subcomponents(model):
    """Replace each Block's forward with sub-component timed version."""
    net = model.net
    blocks = net.blocks
    num_blocks = len(blocks)
    print(f"  Instrumenting {num_blocks} blocks for sub-component timing")

    # Storage for all timing data
    model._subcomp_data = []  # list of per-call data

    from cosmos_policy._src.predict2.networks.minimal_v4_dit import VideoSize
    import torch.amp as amp

    for block in blocks:
        block._subcomp_events = None
        original_forward = block.forward.__func__ if hasattr(block.forward, '__func__') else block.forward

        def make_timed_forward(blk):
            def timed_forward(self, x_B_T_H_W_D, emb_B_T_D, crossattn_emb,
                              rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=None,
                              extra_per_block_pos_emb=None, kv_cache_cfg=None):
                evt = lambda: torch.cuda.Event(enable_timing=True)
                adaln_s, adaln_e = evt(), evt()
                sa_s, sa_e = evt(), evt()
                ca_s, ca_e = evt(), evt()
                ffn_s, ffn_e = evt(), evt()

                if extra_per_block_pos_emb is not None:
                    x_B_T_H_W_D_local = x_B_T_H_W_D + extra_per_block_pos_emb
                else:
                    x_B_T_H_W_D_local = x_B_T_H_W_D

                # ── AdaLN Modulation ──
                adaln_s.record()
                with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
                    if self.use_adaln_lora:
                        shift_sa, scale_sa, gate_sa = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
                        shift_ca, scale_ca, gate_ca = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
                        shift_ff, scale_ff, gate_ff = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(3, dim=-1)
                    else:
                        shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
                        shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
                        shift_ff, scale_ff, gate_ff = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

                # Reshape for broadcasting
                def _r(t): return rearrange(t, "b t d -> b t 1 1 d").type_as(x_B_T_H_W_D_local)
                shift_sa, scale_sa, gate_sa = _r(shift_sa), _r(scale_sa), _r(gate_sa)
                shift_ca, scale_ca, gate_ca = _r(shift_ca), _r(scale_ca), _r(gate_ca)
                shift_ff, scale_ff, gate_ff = _r(shift_ff), _r(scale_ff), _r(gate_ff)
                adaln_e.record()

                B, T, H, W, D = x_B_T_H_W_D_local.shape

                def _fn(x, norm, scale, shift):
                    return norm(x) * (1 + scale) + shift

                # ── Self-Attention ──
                sa_s.record()
                norm_x = _fn(x_B_T_H_W_D_local, self.layer_norm_self_attn, scale_sa, shift_sa)
                video_size = VideoSize(T=T, H=H, W=W)
                if self.cp_size is not None and self.cp_size > 1:
                    video_size = VideoSize(T=T * self.cp_size, H=H, W=W)
                sa_out = rearrange(
                    self.self_attn(rearrange(norm_x, "b t h w d -> b (t h w) d"), None,
                                   rope_emb=rope_emb_L_1_1_D, video_size=video_size, kv_cache_cfg=kv_cache_cfg),
                    "b (t h w) d -> b t h w d", t=T, h=H, w=W)
                x_B_T_H_W_D_local = x_B_T_H_W_D_local + gate_sa * sa_out
                sa_e.record()

                # ── Cross-Attention ──
                ca_s.record()
                norm_x = _fn(x_B_T_H_W_D_local, self.layer_norm_cross_attn, scale_ca, shift_ca)
                ca_out = rearrange(
                    self.cross_attn(rearrange(norm_x, "b t h w d -> b (t h w) d"), crossattn_emb,
                                     rope_emb=rope_emb_L_1_1_D),
                    "b (t h w) d -> b t h w d", t=T, h=H, w=W)
                x_B_T_H_W_D_local = ca_out * gate_ca + x_B_T_H_W_D_local
                ca_e.record()

                # ── FFN/MLP ──
                ffn_s.record()
                norm_x = _fn(x_B_T_H_W_D_local, self.layer_norm_mlp, scale_ff, shift_ff)
                ff_out = self.mlp(norm_x)
                x_B_T_H_W_D_local = x_B_T_H_W_D_local + gate_ff * ff_out
                ffn_e.record()

                self._subcomp_events = {
                    "adaln": (adaln_s, adaln_e),
                    "self_attn": (sa_s, sa_e),
                    "cross_attn": (ca_s, ca_e),
                    "ffn": (ffn_s, ffn_e),
                }
                return x_B_T_H_W_D_local

            return timed_forward

        block.forward = types.MethodType(make_timed_forward(block), block)

    # Instrument denoise to collect per-step data
    original_denoise = model.denoise

    def instrumented_denoise(*args, **kwargs):
        result = original_denoise(*args, **kwargs)
        step_data = []
        for block in blocks:
            if block._subcomp_events is not None:
                step_data.append(block._subcomp_events)
                block._subcomp_events = None
        if step_data:
            model._subcomp_data.append(step_data)
        return result

    model.denoise = instrumented_denoise
    return True


def collect_subcomp_timings(model):
    """Collect sub-component timings after synchronize()."""
    if not model._subcomp_data:
        return None

    num_steps = len(model._subcomp_data)
    num_blocks = len(model._subcomp_data[0]) if model._subcomp_data else 0
    comp_names = ["adaln", "self_attn", "cross_attn", "ffn"]

    # Aggregate across all steps and blocks
    totals = {c: 0.0 for c in comp_names}
    per_step = []

    for step_idx, step_blocks in enumerate(model._subcomp_data):
        step_totals = {c: 0.0 for c in comp_names}
        for block_events in step_blocks:
            for comp in comp_names:
                s, e = block_events[comp]
                t = s.elapsed_time(e)
                totals[comp] += t
                step_totals[comp] += t
        per_step.append(step_totals)

    model._subcomp_data = []

    return {
        "num_steps": num_steps,
        "num_blocks": num_blocks,
        "totals_ms": totals,
        "per_step": per_step,
    }


def run_benchmark(model, cfg, obs, task_label, dataset_stats, num_steps, num_warmup, num_measure):
    from cosmos_policy.experiments.robot.cosmos_utils import get_action
    cfg.num_denoising_steps_action = num_steps

    print(f"\n{'='*60}")
    print(f"  Sub-Component Breakdown: {num_steps} steps")
    print(f"{'='*60}")

    # Warmup
    print(f"  Warmup ({num_warmup})...")
    for i in range(num_warmup):
        with torch.inference_mode():
            get_action(cfg, model, dataset_stats, obs, task_label, seed=i,
                       num_denoising_steps_action=num_steps,
                       generate_future_state_and_value_in_parallel=False)
        torch.cuda.synchronize()
        model._subcomp_data = []

    # Measure
    print(f"  Measuring ({num_measure})...")
    all_results = []
    all_e2e = []

    for i in range(num_measure):
        torch.cuda.synchronize()
        model._subcomp_data = []

        evt = lambda: torch.cuda.Event(enable_timing=True)
        e2e_s, e2e_e = evt(), evt()
        e2e_s.record()

        with torch.inference_mode():
            get_action(cfg, model, dataset_stats, obs, task_label, seed=42,
                       num_denoising_steps_action=num_steps,
                       generate_future_state_and_value_in_parallel=False)

        e2e_e.record()
        torch.cuda.synchronize()

        e2e_ms = e2e_s.elapsed_time(e2e_e)
        all_e2e.append(e2e_ms)

        sc = collect_subcomp_timings(model)
        if sc:
            all_results.append(sc)

        if (i + 1) % 10 == 0 and sc:
            t = sc["totals_ms"]
            total = sum(t.values())
            print(f"    [{i+1}/{num_measure}] e2e={e2e_ms:.0f}ms "
                  f"sa={t['self_attn']:.0f} ca={t['cross_attn']:.0f} ffn={t['ffn']:.0f} "
                  f"adaln={t['adaln']:.0f} sum={total:.0f}ms")

    # Aggregate stats
    comp_names = ["adaln", "self_attn", "cross_attn", "ffn"]
    agg = {}
    for c in comp_names:
        vals = [r["totals_ms"][c] for r in all_results]
        agg[c] = {"mean_ms": float(np.mean(vals)), "std_ms": float(np.std(vals))}

    block_sum = sum(agg[c]["mean_ms"] for c in comp_names)
    e2e_mean = float(np.mean(all_e2e))

    result = {
        "num_steps": num_steps,
        "gpu_e2e_ms": {"mean": e2e_mean, "std": float(np.std(all_e2e))},
        "dit_block_sum_ms": {"mean": block_sum},
        "non_dit_ms": {"mean": e2e_mean - block_sum},
        "components": agg,
        "per_step_example": all_results[0]["per_step"] if all_results else None,
    }

    # Print
    print(f"\n  {'─'*58}")
    print(f"    {'GPU E2E':24s}: {e2e_mean:8.1f} ms")
    print(f"    {'DiT Blocks Total':24s}: {block_sum:8.1f} ms ({block_sum/e2e_mean*100:.1f}%)")
    print(f"    {'Non-DiT':24s}: {e2e_mean-block_sum:8.1f} ms ({(e2e_mean-block_sum)/e2e_mean*100:.1f}%)")
    print(f"    {'─'*54}")
    for c in comp_names:
        pct_e2e = agg[c]["mean_ms"] / e2e_mean * 100
        pct_dit = agg[c]["mean_ms"] / block_sum * 100 if block_sum > 0 else 0
        print(f"    {c:24s}: {agg[c]['mean_ms']:8.1f} ± {agg[c]['std_ms']:5.1f} ms "
              f"({pct_e2e:5.1f}% E2E, {pct_dit:5.1f}% DiT)")
    print(f"  {'─'*58}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, nargs="+", default=[5])
    parser.add_argument("--num-warmup", type=int, default=10)
    parser.add_argument("--num-measure", type=int, default=30)
    parser.add_argument("--experiment-name", default="dit_subcomponents")
    args = parser.parse_args()

    cfg = build_cfg()
    print(f"{'='*60}")
    print(f"  DiT Sub-Component Breakdown")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Steps: {args.num_steps}")
    print(f"{'='*60}")

    from cosmos_policy.experiments.robot.cosmos_utils import (
        get_model, init_t5_text_embeddings_cache, get_t5_embedding_from_cache)

    print("\nLoading model...")
    t0 = time.perf_counter()
    model, _ = get_model(cfg)
    print(f"Model loaded in {time.perf_counter()-t0:.1f}s")

    print("Instrumenting sub-components...")
    instrument_block_subcomponents(model)

    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    _ = get_t5_embedding_from_cache("pick up the black bowl on the stove and place it on the plate")
    task_label = "pick up the black bowl on the stove and place it on the plate"

    obs = create_obs()
    dataset_stats = {"actions_min": np.full(7, -1.0), "actions_max": np.full(7, 1.0),
                     "proprio_min": np.full(9, -1.0), "proprio_max": np.full(9, 1.0)}

    gp = torch.cuda.get_device_properties(0)
    metadata = {
        "gpu_name": gp.name, "vram_gb": round(gp.total_memory / (1024**3), 1),
        "compute_capability": [gp.major, gp.minor],
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "num_warmup": args.num_warmup, "num_measure": args.num_measure,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    all_results = {}
    for ns in args.num_steps:
        all_results[f"steps_{ns}"] = run_benchmark(
            model, cfg, obs, task_label, dataset_stats,
            ns, args.num_warmup, args.num_measure)

    output_dir = os.path.join("benchmark", "results", args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "dit_subcomponent_results.json")
    with open(path, "w") as f:
        json.dump({"results": all_results, "metadata": metadata}, f, indent=2)
    print(f"\nResults saved to {path}")

    # Final comparison
    print(f"\n{'='*80}")
    print(f"  SUB-COMPONENT COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Steps':>5} | {'E2E':>8} | {'Self-Attn':>10} | {'Cross-Attn':>10} | {'FFN':>10} | {'AdaLN':>8} | {'Non-DiT':>8}")
    print(f"  {'─'*5}-+-{'─'*8}-+-{'─'*10}-+-{'─'*10}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}")
    for ns in args.num_steps:
        r = all_results[f"steps_{ns}"]
        c = r["components"]
        print(f"  {ns:5d} | {r['gpu_e2e_ms']['mean']:7.0f}ms"
              f" | {c['self_attn']['mean_ms']:8.0f}ms"
              f" | {c['cross_attn']['mean_ms']:8.0f}ms"
              f" | {c['ffn']['mean_ms']:8.0f}ms"
              f" | {c['adaln']['mean_ms']:6.0f}ms"
              f" | {r['non_dit_ms']['mean']:6.0f}ms")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
