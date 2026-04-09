# Cosmos Policy: Denoising Step Reduction Experiment Summary

> Date: 2026-03-23~24
> GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB)
> Benchmark: LIBERO-Spatial (10 tasks x 50 episodes = 500 trials, seed 195, deterministic)

---

## 1. Slot Convergence Analysis (Exp 0)

Different slot types converge at different rates during denoising.

| Step | Action MSE | Camera MSE | Value MSE | Action CosSim |
|------|-----------|-----------|----------|---------------|
| 1 | 0.006171 | 0.002245 | 0.188949 | 0.979 |
| 3 | 0.000060 | 0.000090 | 0.007170 | 0.9998 |
| 5 | 0.000019 | 0.000038 | 0.000180 | 0.9999 |
| 9 | 0.000014 | 0.000027 | 0.000018 | 0.9999 |

- Value slot converges **84x slower** than camera at step 1
- Action converges slightly faster than camera
- 3 steps: action MSE = 0.00003 (effectively zero)

---

## 2. Latency Benchmark (Exp 1)

| Steps | GPU E2E (ms) | Denoise (ms) | VAE+Cond (ms) | Speedup vs 5-step | Paper H100 (ms) |
|-------|-------------|-------------|---------------|-------------------|-----------------|
| 1 | 146.7 | 64.5 | 69.3 | **1.96x** | 160 |
| 3 | 243.7 | 163.7 | 68.3 | 1.18x | N/A |
| 5 | 287.4 | 209.2 | 67.3 | 1.00x | 610 |

RTX PRO 6000 Blackwell is ~2.1x faster than H100 at 5-step inference.
VAE+conditioning overhead (~68ms) is fixed regardless of denoising steps.

---

## 3. LIBERO-Spatial Accuracy vs Denoising Steps

### Per-Task Results (10 tasks x 50 episodes each)

| Task | 5-step (ours) | 3-step (ours) | 1-step (ours) |
|------|:------------:|:------------:|:------------:|
| 1 | 98% | 98% | **100%** |
| 2 | 98% | 100% | **100%** |
| 3 | 100% | 100% | **100%** |
| 4 | 94% | 94% | **98%** |
| 5 | 100% | 100% | **98%** |
| 6 | 90% | 90% | **94%** |
| 7 | 98% | 98% | **98%** |
| 8 | 94% | 94% | **96%** |
| 9 | 98% | 98% | **100%** |
| 10 | 100% | 100% | **100%** |
| **Total** | **97.0%** | **97.2%** | **98.4%** |

### Comparison with Paper

| Steps | Latency (ms) | Speedup | Ours (RTX PRO 6000) | Paper (H100) | Delta vs Paper |
|-------|-------------|---------|--------------------|--------------|--------------:|
| 1 | 147 | **1.96x** | **98.4%** | N/A (LIBERO) | — |
| 3 | 244 | 1.18x | **97.2%** | N/A | — |
| 5 | 287 | 1.00x | 97.0% | **98.1%** | -1.1% |

**Paper (H100) RoboCasa reference:**

| Steps | Latency (ms) | RoboCasa SR |
|-------|-------------|------------|
| 1 | 160 | 66.4% |
| 5 | 610 | **67.1%** |

---

## 4. Pareto Summary: Latency vs Accuracy (LIBERO-Spatial)

| Method | Latency | Speedup | SR (%) | Notes |
|--------|---------|---------|--------|-------|
| Paper 5-step (H100) | 610ms | 1.00x | **98.1** | 3 seeds avg |
| Paper 1-step (H100) | 160ms | 3.81x | N/A | LIBERO not reported |
| Ours 5-step | 287ms | 1.00x | 97.0 | seed 195 |
| Ours 3-step | 244ms | 1.18x | 97.2 | **lossless vs 5-step** |
| **Ours 1-step** | **147ms** | **1.96x** | **98.4** | **+1.4% vs 5-step!** |

### Key Finding

1-step achieves **98.4%** vs 5-step's 97.0% on LIBERO-Spatial — **+1.4% higher** while being **1.96x faster**.
This is consistent with the paper's RoboCasa finding (1-step 66.4% vs 5-step 67.1%, only -0.7% drop).

The asymmetry: action slots converge very fast (CosSim=0.979 at step 1), so reducing denoising steps
barely hurts action quality. The value/future-state slots are denoising targets but not used for control.

---

## 5. RoboCasa Step Sweep (Exp 4) — IN PROGRESS

> Running: 1-step and 3-step sweeps (24 tasks x 50 trials each)
> 5-step reproduction result already available: **65.6%** (paper: 67.1%)

| Steps | Latency | Ours (RTX PRO 6000) | Paper (H100) |
|-------|---------|--------------------|--------------| 
| 1 | ~147ms | TBD | **66.4%** |
| 3 | ~244ms | TBD | N/A |
| 5 | ~287ms | **65.6%** (repro) | **67.1%** |
