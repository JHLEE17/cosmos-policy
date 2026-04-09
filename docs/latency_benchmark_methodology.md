# Cosmos Policy Latency Benchmark Methodology

Cosmos Policy의 inference latency를 정확하고 재현 가능하게 측정하기 위한 방법론입니다.

---

## 1. 개요

### 목적

1. **병목 구간 식별**: pipeline 중 어떤 component가 가장 느린지 파악
2. **최적화 효과 검증**: precision/backend/denoising steps 변경 전후 비교
3. **깊은 분석**: DiT 내부 sub-component까지 breakdown
4. **Hardware 간 비교**: 동일 모델을 다른 GPU에서 측정하여 성능 비교

### Cosmos Policy Pipeline 구성

```
Direct Policy (Planning 없이):
  ┌─ CPU ─────────────────────────────────────────────┐
  │ Image Preprocessing (resize, augment, normalize)  │
  │ Pseudo-video Assembly (duplicate, stack)           │
  │ Proprio Normalization                              │
  │ T5 Embedding Loading (.pkl)                        │
  └───────────────────────────────────────────────────┘
  ┌─ GPU ─────────────────────────────────────────────┐
  │ VAE Encode (Wan2.1)                                │
  │ Latent Injection (proprio into gt_frames)          │
  │ Conditioning Mask Setup                            │
  │ Denoising Loop (5~10 × DiT forward)  ← 주요 병목  │
  │ Action Extraction (flatten, avg, unnorm)           │
  └───────────────────────────────────────────────────┘

Planning Mode (추가):
  ┌─ GPU ─────────────────────────────────────────────┐
  │ Action Proposal × N (Denoising Loop × N)           │
  │ Future State Prediction × 3 (Denoising Loop × 3)   │
  │ Value Prediction × 5 (Denoising Loop × 5)          │
  │ Value Aggregation (majority mean)                   │
  └───────────────────────────────────────────────────┘
```

### 핵심 원칙

| 원칙 | 설명 |
|------|------|
| **CUDA Events 사용** | `time.time()` 대신 `torch.cuda.Event`로 GPU kernel 시간을 정확히 측정 |
| **E2E와 Component Breakdown의 sync 전략 분리** | 측정 목적에 따라 다른 sync 전략 사용 (아래 2.2 참조) |
| **E2E vs Component Sum 비교** | 전체 GPU 시간과 개별 component 합을 비교하여 pipeline gap 파악 |
| **Warmup 충분히 수행** | torch.compile JIT 안정화 + CUDA context 워밍업 |
| **Iteration Type 불필요** | Cosmos Policy는 KV cache/streaming 없이 매 query가 독립적 |

---

## 2. 측정 방법론

### 2.1 시간 측정의 3가지 레벨

```
Level 1: Wall-clock (time.perf_counter)
  └─ CPU 스케줄링 + GPU 실행 시간 모두 포함
  └─ 실제 사용자가 체감하는 latency (로봇 제어 주기에 직접 영향)

Level 2: GPU E2E (CUDA Event 1쌍)
  └─ GPU 커널만의 순수 실행 시간
  └─ CPU 오버헤드 제외

Level 3: Component Breakdown (CUDA Event N쌍)
  └─ 각 component별 GPU 시간
  └─ 합산하여 GPU E2E와 비교 → gap = pipeline overhead
```

### 2.2 CUDA Event와 cuda.synchronize()의 올바른 이해

#### CUDA Event의 동작 원리

```python
event = torch.cuda.Event(enable_timing=True)
event.record()  # GPU command stream에 "여기 통과할 때 시간 기록해" 마커 삽입
```

CUDA Event는 GPU의 command stream에 timestamp 마커를 삽입합니다.
- `start.record()` → stream에 시작 마커 삽입
- (GPU 작업 실행)
- `end.record()` → stream에 종료 마커 삽입
- `start.elapsed_time(end)` → 두 마커 사이의 GPU 시간 반환

**`cuda.synchronize()`는 event 기록의 정확성과 무관합니다.**
동일 stream에서 순차 실행되는 작업이면, event만으로 정확한 시간을 측정할 수 있습니다.
`synchronize()`는 **event 값을 CPU에서 읽기 전**에 GPU 작업이 완료되었음을 보장하기 위해 필요합니다.

#### E2E 측정 vs Component Breakdown 측정

```
E2E 측정 목적: "실제 환경에서 전체 pipeline이 얼마나 걸리는가?"
  → 중간 sync 없이 측정 (GPU pipeline이 자연스럽게 동작)
  → cuda.synchronize()는 맨 마지막에 1번만

Component Breakdown 목적: "각 component의 독립적 비용은 얼마인가?"
  → 두 가지 선택지가 있음:
```

**Option A: 중간 sync 없이 CUDA Event만 사용 (권장 기본값)**

```python
comp_a_start.record()
output_a = component_a(inputs)
comp_a_end.record()

comp_b_start.record()
output_b = component_b(output_a)
comp_b_end.record()

torch.cuda.synchronize()  # 마지막에 1번
```

- 장점: GPU pipeline이 자연스럽게 동작하므로 실제 실행과 동일한 조건
- 장점: component 간 kernel overlap이 있으면 그대로 반영됨
- 주의: component_sum ≤ gpu_e2e (overlap이 있으면 합이 더 작을 수 있음)
- **동일 CUDA stream에서 순차 실행되는 경우**, overlap 없이 정확한 시간 측정 가능

**Option B: 각 component 사이에 sync 삽입 (독립 비용 측정)**

```python
comp_a_start.record()
output_a = component_a(inputs)
comp_a_end.record()
torch.cuda.synchronize()  # component A 완료 대기

comp_b_start.record()
output_b = component_b(output_a)
comp_b_end.record()
torch.cuda.synchronize()  # component B 완료 대기
```

- 장점: 각 component의 **독립적 비용**을 정확히 측정
- 단점: sync가 GPU pipeline을 끊어 실제 실행보다 느리게 측정될 수 있음
- 용도: "이 component를 최적화하면 얼마나 빨라지나?" 판단 시 유용

**Cosmos Policy 권장**: Option A를 기본으로 사용하되, 특정 component의 독립 비용이 필요할 때만 Option B로 재측정. Cosmos Policy의 inference는 단일 CUDA stream에서 순차 실행되므로 Option A로도 정확한 component 시간 측정이 가능합니다.

### 2.3 CUDA Event 기반 측정 패턴

```python
import torch
import time

def forward_with_timing(model, data_batch, num_denoising_steps):
    # ── Wall-clock 시작 ──
    torch.cuda.synchronize()
    wall_start = time.perf_counter()

    # ── GPU E2E 이벤트 ──
    gpu_e2e_start = torch.cuda.Event(enable_timing=True)
    gpu_e2e_end = torch.cuda.Event(enable_timing=True)
    gpu_e2e_start.record()

    # ── 1. Image Preprocessing (CPU) ──
    # (CUDA event로 측정 불가 - wall-clock으로 별도 측정)

    # ── 2. VAE Encode ──
    vae_s, vae_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    vae_s.record()
    latent_state = model.tokenizer.encode(data_batch["video"])
    vae_e.record()

    # ── 3. Conditioning Setup (latent injection + mask) ──
    cond_s, cond_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    cond_s.record()
    x0_fn = model.get_x0_fn_from_batch(data_batch, guidance=0)
    cond_e.record()

    # ── 4. Denoising Loop ──
    denoise_s, denoise_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    denoise_s.record()
    generated_latent = model.sampler(x0_fn, x_sigma_max, num_steps=num_denoising_steps,
                                      sigma_max=80, sigma_min=4)
    denoise_e.record()

    # ── 5. Action Extraction ──
    extract_s, extract_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    extract_s.record()
    actions = extract_action_chunk_from_latent_sequence(generated_latent, action_shape, action_indices)
    extract_e.record()

    # ── GPU E2E 종료 ──
    gpu_e2e_end.record()
    torch.cuda.synchronize()  # 여기서 1번만 sync (event 값 읽기 전)

    # ── 시간 계산 ──
    timing = {
        "wall_total": time.perf_counter() - wall_start,
        "gpu_e2e": gpu_e2e_start.elapsed_time(gpu_e2e_end) / 1000,
        "vae_encode": vae_s.elapsed_time(vae_e) / 1000,
        "conditioning_setup": cond_s.elapsed_time(cond_e) / 1000,
        "denoising_loop": denoise_s.elapsed_time(denoise_e) / 1000,
        "action_extraction": extract_s.elapsed_time(extract_e) / 1000,
    }
    timing["component_sum"] = sum(v for k, v in timing.items()
                                   if k not in ["wall_total", "gpu_e2e", "component_sum"])
    return actions, timing
```

### 2.4 E2E vs Component Sum 비교의 의미

```
GPU E2E:        ████████████████████████████████████  950ms
Component Sum:  ███████████████████████████████████   945ms
Gap:            ·····                                   5ms

Wall-clock:     ██████████████████████████████████████ 980ms
CPU Overhead:   ··                                     30ms
```

| 비교 항목 | 의미 |
|-----------|------|
| `gpu_e2e - component_sum` | Component 사이의 GPU idle gap (pipeline 비효율) |
| `wall_clock - gpu_e2e` | CPU overhead (Python, numpy 변환, data transfer 등) |

---

## 3. Cosmos Policy Pipeline Components

### 3.1 Component 정의

| Component | 위치 | 예상 비율 | 설명 |
|-----------|------|-----------|------|
| `image_preprocess` | CPU | <1% | resize, augment, numpy 변환 |
| `pseudo_video_assembly` | CPU→GPU | <1% | 이미지 복제, 시퀀스 구성, GPU 전송 |
| `t5_embedding_load` | CPU→GPU | <1% | .pkl에서 text embedding 로드 |
| `vae_encode` | GPU | ~5-10% | Wan2.1 VAE encoder |
| `conditioning_setup` | GPU | ~2-5% | latent injection, mask 구성, x0_fn 생성 |
| **`denoising_loop`** | **GPU** | **~80-90%** | **5~10 × DiT forward pass (주요 병목)** |
| `action_extraction` | GPU→CPU | <1% | flatten, average, unnormalize |

### 3.2 Denoising Loop 내부 (Deep Breakdown 대상)

```
denoising_loop (5~10 steps)
  └─ 매 step:
       ├─ sigma schedule 계산
       ├─ DiT forward pass ← 대부분의 시간
       │    ├─ Patch Embed (18ch → 2048)
       │    ├─ 28 Transformer Blocks
       │    │    ├─ Self-Attention (예상 ~45%)
       │    │    ├─ Cross-Attention (예상 ~23%)
       │    │    ├─ FFN (예상 ~28%)
       │    │    └─ Overhead: LayerNorm, residual, AdaLN (~4%)
       │    └─ Final Layer + Unpatchify
       ├─ EDM preconditioning (c_skip, c_out 적용)
       └─ Conditional frame 교체 (gt_frames로 복원)
```

### 3.3 Planning Mode 추가 Components

| Component | 설명 | 횟수 |
|-----------|------|------|
| `action_proposal` | N개 action 후보 denoising (10 steps each) | N (기본 8) |
| `future_state_pred` | 미래 상태 denoising (5 steps each) | 3 × N |
| `value_pred` | Value denoising (5 steps each) | 5 × 3 × N |
| `value_aggregation` | Majority mean 계산 | 1 |

---

## 4. Benchmark Script 구조

### 4.1 기본 구조

```python
# benchmark_latency.py

def run_benchmark(args):
    # 1. 모델 로딩
    model = get_model(cfg)
    model.eval()

    # 2. Synthetic 입력 생성
    #    (실제 환경과 동일한 shape의 dummy 데이터)
    def create_synthetic_input():
        images = torch.randn(B, 3, 224, 224).cuda()         # camera images
        proprio = torch.randn(B, PROPRIO_DIM).cuda()          # joint angles
        text_emb = torch.randn(B, 512, 1024).cuda()          # T5 embedding
        return build_data_batch(images, proprio, text_emb)

    # 3. Warmup
    for i in range(args.num_warmup):
        torch.cuda.synchronize()
        _ = forward_with_timing(model, create_synthetic_input(), args.num_steps)

    # 4. 측정
    all_timings = []
    for i in range(args.num_measure):
        torch.cuda.synchronize()
        _, timing = forward_with_timing(model, create_synthetic_input(), args.num_steps)
        all_timings.append(timing)

    # 5. 통계 계산 + 저장
    results = compute_statistics(all_timings)
    save_results(results, args.output_dir)
```

### 4.2 Cosmos Policy 특성: Iteration Type 불필요

DreamZero와 달리 Cosmos Policy는:
- KV cache 없음
- Streaming 모드 없음
- 매 query가 완전히 독립적 (이전 query의 상태를 재사용하지 않음)

따라서 `full_rebuild / reset_frame / incremental` 같은 iteration type 분류가 **불필요**합니다.
모든 iteration이 동일한 조건이므로 단순 평균/표준편차로 통계를 계산합니다.

```python
def compute_statistics(all_timings):
    stats = {}
    for key in all_timings[0].keys():
        values = [t[key] * 1000 for t in all_timings]  # s → ms
        stats[key] = {
            "mean_ms": np.mean(values),
            "std_ms": np.std(values),
            "min_ms": np.min(values),
            "max_ms": np.max(values),
            "median_ms": np.median(values),
        }
    return stats
```

### 4.3 Warmup 설계

| 항목 | 권장값 | 이유 |
|------|--------|------|
| num_warmup | 10+ | CUDA context 안정화 + 첫 실행 JIT overhead 제거 |
| num_measure | 30+ | 충분한 샘플로 std 안정화 |

> Cosmos Policy는 기본적으로 torch.compile을 사용하지 않으므로 DreamZero보다 warmup이 적게 필요합니다.
> 다만 CUDA lazy initialization, cuDNN auto-tuning 등을 위해 최소 10회는 필요합니다.

### 4.4 측정 시 주의사항

```python
# ❌ 잘못된 측정: 매 iteration 사이에 불필요한 sync
for i in range(num_measure):
    torch.cuda.synchronize()  # 이건 OK (이전 iteration 완료 보장)
    _, timing = forward_with_timing(...)
    all_timings.append(timing)
    torch.cuda.synchronize()  # 이건 불필요 (forward 내부에서 이미 sync함)

# ✅ 올바른 측정
for i in range(num_measure):
    torch.cuda.synchronize()  # iteration 시작 전 clean state 보장
    _, timing = forward_with_timing(...)  # 내부에서 마지막에 sync
    all_timings.append(timing)
```

---

## 5. Deep Breakdown 방법론

### 5.1 개념

일반 benchmark는 pipeline-level component만 측정합니다 (VAE, conditioning, denoising 등).
Deep breakdown은 **denoising loop 내부**를 추가로 분석합니다.

```
Pipeline Level (기본):
  Image Prep → VAE Encode → Conditioning → [Denoising Loop] → Action Extract
                                             ↑ ~85% 차지

Denoising Step Level:
  [Denoising Loop]
    └─ Step 1: DiT forward (σ=80)
    └─ Step 2: DiT forward (σ=44)
    └─ ...
    └─ Step 5: DiT forward (σ=4)    ← step별 시간 차이 확인

DiT Internal Level:
  [DiT Forward]
    └─ Patch Embed + Pos Embed (~1%)
    └─ 28 Transformer Blocks (~98%)  ← 여기를 더 깊게
    └─ Final Layer + Unpatchify (~1%)

Block Level (deepest):
  [Blocks × 28]
    └─ Self-Attention (~45%)
    └─ Cross-Attention (~23%)
    └─ FFN (~28%)
    └─ Overhead (~4%)  ← LayerNorm, residual, AdaLN 등
```

### 5.2 Denoising Step별 측정

```python
# CosmosPolicySampler._forward_impl() 내부에 계측 삽입

def _forward_impl_with_timing(self, denoiser_fn, noisy_input, sampler_cfg, num_steps):
    step_timings = []

    for i, sigma in enumerate(sigmas):
        step_s = torch.cuda.Event(enable_timing=True)
        step_e = torch.cuda.Event(enable_timing=True)
        step_s.record()

        # 실제 denoising step
        denoised = denoiser_fn(x_t, sigma)

        step_e.record()
        step_timings.append({
            "step": i,
            "sigma": sigma.item(),
            "events": (step_s, step_e),
        })

    return denoised, step_timings
```

### 5.3 Block-level 계측 패턴

```python
# MinimalV1LVGDiT 또는 MiniTrainDIT 내부에 계측 삽입

class InstrumentedTransformerBlock(nn.Module):
    def forward(self, x, context, timestep_emb):
        _breakdown = getattr(self, '_deep_breakdown', False)
        if _breakdown:
            _evt = lambda: torch.cuda.Event(enable_timing=True)
            sa_s, sa_e = _evt(), _evt()
            ca_s, ca_e = _evt(), _evt()
            ff_s, ff_e = _evt(), _evt()

        # Self-Attention
        if _breakdown: sa_s.record()
        x = x + self.self_attn(self.norm1(x))
        if _breakdown: sa_e.record()

        # Cross-Attention
        if _breakdown: ca_s.record()
        x = x + self.cross_attn(self.norm2(x), context)
        if _breakdown: ca_e.record()

        # FFN
        if _breakdown: ff_s.record()
        x = x + self.ffn(self.norm3(x))
        if _breakdown: ff_e.record()

        if _breakdown:
            self._last_timing = {
                'self_attn': (sa_s, sa_e),
                'cross_attn': (ca_s, ca_e),
                'ffn': (ff_s, ff_e),
            }
        return x
```

### 5.4 torch.compile과의 충돌 해결

```python
# torch.compile로 감싸진 함수 내부에는 CUDA event를 삽입할 수 없음
# (compiled graph는 동적 event recording을 허용하지 않음)

# 해결: 환경변수로 breakdown 모드를 제어
import os

DEEP_BREAKDOWN = os.environ.get("DEEP_BREAKDOWN", "false").lower() == "true"

if DEEP_BREAKDOWN:
    # torch.compile 비활성화 + 내부 계측 활성화
    print("Deep breakdown enabled: torch.compile disabled")
else:
    # torch.compile 적용 (있다면)
    pass
```

> **중요**: Deep breakdown 결과는 compile 비활성화 상태이므로 **절대 시간은 실제보다 느릴 수 있습니다.** 하지만 **component 간 비율**은 유효합니다.

---

## 6. 결과 구조 및 JSON 스키마

### 6.1 JSON 출력 형식

```jsonc
{
  "mean_ms": {
    // Pipeline level
    "wall_total": 980.0,
    "gpu_e2e": 950.0,
    "component_sum": 945.0,
    "vae_encode": 55.0,
    "conditioning_setup": 30.0,
    "denoising_loop": 850.0,
    "action_extraction": 10.0,

    // Denoising step breakdown (--step-breakdown 시)
    "denoise_step_0": 172.0,
    "denoise_step_1": 170.0,
    "denoise_step_2": 170.0,
    "denoise_step_3": 169.0,
    "denoise_step_4": 169.0,

    // DiT internal breakdown (--dit-breakdown 시)
    "dit_patch_embed": 12.0,
    "dit_blocks": 835.0,
    "dit_final_layer": 3.0,

    // Block-level breakdown
    "dit_self_attn": 375.0,
    "dit_cross_attn": 192.0,
    "dit_ffn": 234.0,
    "dit_block_overhead": 34.0
  },
  "std_ms": { /* 동일 키 구조 */ },
  "count": 30,
  "raw_timings": [
    { /* iteration별 개별 측정값 */ }
  ],
  "metadata": {
    "gpu_name": "NVIDIA H100",
    "num_gpus": 1,
    "suite": "libero",
    "state_t": 9,
    "num_denoising_steps": 5,
    "action_dim": 7,
    "chunk_size": 16,
    "sigma_max": 80,
    "sigma_min": 4,
    "solver": "2ab",
    "batch_size": 1,
    "num_warmup": 10,
    "num_measure": 30,
    "timestamp": "2026-03-19T12:00:00",
    "libraries": {
      "torch": "2.x.x",
      "cuda": "12.x"
    }
  }
}
```

### 6.2 디렉토리 구조

```
benchmark/
├── benchmark_latency.py           # 메인 벤치마크 스크립트
├── visualize_latency.py           # Pipeline-level 시각화
├── visualize_dit_breakdown.py     # Deep breakdown 시각화
├── compare_experiments.py         # 실험간 비교
└── results/
    ├── {experiment_name}/
    │   ├── latency_results.json
    │   ├── latency_breakdown.png
    │   └── dit_deep_breakdown.png
    └── comparison_summary.md
```

---

## 7. Visualization

### 7.1 Pipeline-level

| 패널 | 내용 |
|------|------|
| Stacked Bar | Component별 latency (ms) + error bar |
| Pie Chart | Component별 비율 (%) |

```python
COMPONENT_CONFIG = {
    "vae_encode":          {"label": "VAE Encode",          "color": "#C44E52"},
    "conditioning_setup":  {"label": "Conditioning Setup",  "color": "#8172B2"},
    "denoising_loop":      {"label": "Denoising Loop",      "color": "#CCB974"},
    "action_extraction":   {"label": "Action Extraction",   "color": "#64B5CD"},
}
```

### 7.2 Deep Breakdown

| 패널 | 내용 |
|------|------|
| Pipeline Breakdown | 전체 pipeline stacked horizontal bar |
| Per-Step Timeline | 각 denoising step의 latency (bar chart) |
| DiT Internal | Patch Embed / Blocks / Final Layer |
| Block Components | Self-Attn / Cross-Attn / FFN (pie chart) |

---

## 8. 실험 비교

### 8.1 Cosmos Policy에서 비교할 변수

| 변수 | 비교 항목 |
|------|-----------|
| Denoising steps | 1 vs 5 vs 10 |
| Suite | LIBERO (T=9) vs ALOHA (T=11) |
| Batch size | 1 vs N (planning의 best-of-N) |
| Solver | 2ab vs 다른 solver |
| Parallel vs Autoregressive | Direct policy vs Planning mode |
| GPU | H100 vs A100 vs RTX 4090 |

### 8.2 실행 예시

```bash
# Direct Policy, LIBERO, 5 steps
python benchmark/benchmark_latency.py \
    --suite libero \
    --num-steps 5 \
    --num-warmup 10 \
    --num-measure 30 \
    --experiment-name libero_5steps

# Direct Policy, ALOHA, 10 steps
python benchmark/benchmark_latency.py \
    --suite aloha \
    --num-steps 10 \
    --experiment-name aloha_10steps

# Deep breakdown (DiT 내부 분석)
DEEP_BREAKDOWN=true python benchmark/benchmark_latency.py \
    --suite libero \
    --num-steps 5 \
    --experiment-name libero_5steps_breakdown

# 비교 시각화
python benchmark/compare_experiments.py \
    --inputs results/libero_5steps results/aloha_10steps
```

---

## 9. 예상 결과 (논문 Appendix A.4.2 기준)

### 9.1 Direct Policy Latency

| Suite | Denoising Steps | 예상 GPU E2E | 예상 Wall-clock |
|-------|----------------|-------------|----------------|
| LIBERO | 5 | ~550ms | ~610ms |
| RoboCasa | 5 | ~550ms | ~610ms |
| ALOHA | 10 | ~880ms | ~950ms |
| LIBERO | 1 | ~130ms | ~160ms |

> 논문 보고 수치: 5 steps=0.61s, 10 steps=0.95s, 1 step=0.16s (1×H100 기준)
> Denoising loop이 전체의 ~85-90%를 차지할 것으로 예상

### 9.2 Planning Mode Latency (ALOHA, 8 GPU 병렬)

| Phase | Steps/call | Calls | 예상 시간 |
|-------|-----------|-------|----------|
| Action Proposals (×8) | 10 | 8 (병렬) | ~950ms |
| Future State (×3 per action) | 5 | 3 | ~900ms |
| Value (×5 per state) | 5 | 15 | ~2250ms |
| Aggregation | - | 1 | <10ms |
| **Total** | | | **~4.9s** |

---

## 10. 체크리스트

- [ ] CUDA Event는 `enable_timing=True`로 생성되었는가?
- [ ] GPU E2E event pair가 모든 GPU 작업을 감싸는가?
- [ ] `cuda.synchronize()`는 event 값을 읽기 전에 호출되는가?
- [ ] Component event가 겹치지 않게 순차적으로 배치되었는가?
- [ ] Warmup 횟수가 충분한가? (최소 10회)
- [ ] 측정 횟수가 통계적으로 유의미한가? (최소 30회)
- [ ] Metadata에 hardware/software/suite 정보가 포함되었는가?
- [ ] Deep breakdown 시 torch.compile이 비활성화되는가?
- [ ] Synthetic input이 실제 inference와 동일한 shape인가?
- [ ] Planning mode 측정 시 GPU 수와 병렬화 방식이 명시되었는가?
