# Cosmos Policy: Model Architecture & Inference Flow 상세 분석

> 이 문서는 Cosmos Policy 논문 및 공식 코드베이스를 기반으로 모델 아키텍처, 데이터 흐름, 차원 정보를 상세히 정리한 것입니다.
> Flow chart 작성을 위한 참조 문서로 활용할 수 있습니다.

---

## 1. 전체 개요 (High-Level Overview)

Cosmos Policy는 NVIDIA의 **Cosmos-Predict2-2B** 비디오 생성 모델을 **아키텍처 변경 없이** 로봇 정책(policy)으로 파인튜닝한 모델입니다.

핵심 아이디어: 로봇의 action, proprioception, value를 **latent frame으로 인코딩**하여 비디오 모델의 latent diffusion sequence에 직접 삽입(Latent Frame Injection)합니다.

### 중요 개념: "Pseudo-Video" 입력
Cosmos Policy는 실제 temporal history video clip을 입력으로 사용하지 **않습니다**. 대신, **단일 시점(t)의 관측값**으로 구성된 **구조화된 pseudo-video**를 생성합니다:
- 현재 카메라 이미지들과 blank placeholder 이미지들을 조합하여 pseudo-video sequence를 구성
- VAE로 인코딩한 후 placeholder latent frame들을 action/proprio/value 데이터로 overwrite
- 비디오 모델의 backbone을 **multi-slot latent generator**로 활용

즉, "비디오를 보고 action을 출력한다"는 표현은 부분적으로만 정확합니다. 실제로는 현재 관측값 기반의 구조화된 latent sequence에서 빈 slot을 생성(denoising)하는 방식입니다.

### 모델이 동시에 수행하는 3가지 역할:
1. **Policy (정책)**: 현재 상태 s에서 action chunk a 생성 → `p(a, s', V(s') | s)`
2. **World Model (세계 모델)**: 현재 상태 s + action a에서 미래 상태 s' 예측 → `p(s', V(s') | s, a)`
3. **Value Function (가치 함수)**: 미래 상태의 expected return 예측 → `p(V(s') | s, a, s')`

---

## 2. Base Model: Cosmos-Predict2-2B 아키텍처

### 2.1 Diffusion Transformer (DiT) - MinimalV1LVGDiT

| 파라미터 | 2B 모델 (Cosmos Policy 사용) | 7B (참고) | 14B (참고) |
|---|---|---|---|
| **Hidden Dimension (model_channels)** | **2048** | 4096 | 5120 |
| **Transformer Blocks (num_blocks)** | **28** | 28 | 36 |
| **Attention Heads (num_heads)** | **16** | 32 | 40 |
| **Head Dimension** | **128** (= 2048/16) | 128 | 128 |
| **MLP Ratio** | **4.0** (hidden=8192) | 4.0 | 4.0 |
| **Patch Spatial** | **2** (2x2 spatial patch) | 2 | 2 |
| **Patch Temporal** | **1** (1 frame per patch) | 1 | 1 |
| **Input/Output Channels** | **16** (VAE latent channels) | 16 | 16 |
| **Effective Patch Embed Input Channels** | **18** (16+1+1, 아래 참조) | 18 | 18 |
| **Max Input Height (latent)** | 240 | 240 | 240 |
| **Max Input Width (latent)** | 240 | 240 | 240 |
| **Max Frames** | 128 | 128 | 128 |

> 소스: `cosmos_policy/_src/predict2/configs/video2world/defaults/net.py`

### 2.1.1 Patch Embed 입력 채널 상세 (18 channels)

Latent 자체는 16채널이지만, Patch Embedding layer에 실제로 입력되는 채널 수는 **18**입니다:

```
16 (VAE latent channels)
 + 1 (condition_video_input_mask)  ← MinimalV1LVGDiT에서 추가
 + 1 (padding_mask)                ← MiniTrainDIT에서 concat_padding_mask=True일 때 추가
 = 18 channels → Patch Embedding
```

- `MinimalV1LVGDiT.__init__()`: `kwargs["in_channels"] += 1` → condition mask 채널 추가
- `MiniTrainDIT.__init__()`: `in_channels = in_channels + 1 if concat_padding_mask` → padding mask 채널 추가
- 모델 **출력**은 여전히 16채널 (추가 채널은 입력 conditioning 신호일 뿐)

> 소스: `cosmos_policy/_src/predict2/networks/minimal_v1_lvg_dit.py:27`, `minimal_v4_dit.py:1611`

### 2.2 Positional Embedding
- **타입**: RoPE3D (3D Rotary Position Embedding)
- **학습 가능 여부**: True (`pos_emb_learnable=True`)
- **보간 방식**: crop (`pos_emb_interpolation="crop"`)
- **Extrapolation Ratio**: H=1.0, W=1.0, T=1.0
- **AdaLN-LoRA**: 사용 (`adaln_lora_dim=256`)

### 2.3 Attention Mechanism
- **Q/K/V Projection**: Linear(2048 → 2048)
- **QKV Format**: `bshd` (batch, sequence, heads, head_dim)
- **Q, K Normalization**: RMSNorm (head_dim=128)
- **Cross-Attention Context**: T5-XXL text embeddings (dim=1024, max 512 tokens)
- **Backend**: `minimal_a2a` (또는 transformer_engine)

### 2.4 Conditioning
- **Text Encoder**: T5-XXL
- **Text Embedding Dim**: 1024
- **Context Token Number**: 512 (max)
- **Text Conditioning**: Cross-Attention을 통해 주입
- **Noise Level Conditioning**: Adaptive Layer Normalization (AdaLN)으로 sigma 조건부 처리

---

## 3. Video Tokenizer: Wan2.1 Spatiotemporal VAE

### 3.1 압축 비율

| 차원 | 입력 | 출력 (Latent) | 압축 비율 |
|---|---|---|---|
| **Temporal** | N frames | 1 + (N-1)//4 frames | 4:1 (첫 프레임 단독 인코딩) |
| **Height** | H pixels | H/8 pixels | 8:1 |
| **Width** | W pixels | W/8 pixels | 8:1 |
| **Channels** | 3 (RGB) | 16 (latent) | - |

### 3.2 Cosmos Policy에서의 실제 변환

```
입력 이미지: 224 x 224 x 3 (RGB)
    ↓ VAE Encode
Latent Frame: 28 x 28 x 16

하나의 latent frame 크기: C'=16, H'=28, W'=28
총 latent elements per frame: 16 × 28 × 28 = 12,544
```

### 3.3 Temporal 압축 특징
- **첫 번째 프레임**: temporal 압축 없이 단독 인코딩 (image conditioning용)
- **나머지 프레임**: 4프레임 단위로 temporal 압축
- 따라서 각 이미지를 **4장 복제**하여 1개의 latent frame을 생성 (`num_duplicates_per_image=4`)

```
Temporal compression formula: latent_frames = 1 + (num_pixel_frames - 1) // 4

예시 (LIBERO, state_t=9):
  입력 이미지 시퀀스: 1(blank) + 8×4 = 33장 (chunk_duration=33)
       ↓ VAE Encode
  Latent 시퀀스: 1 + (33 - 1) // 4 = 1 + 8 = 9 latent frames (state_t=9)
```

> 소스: `cosmos_policy/tokenizers/wan2pt1.py:540-541`

---

## 4. Latent Frame Injection: 핵심 메커니즘

### 4.1 개념
기존 비디오 모델의 latent sequence에 새로운 modality(action, proprioception, value)를 **latent frame 형태로 직접 삽입**합니다. 아키텍처 변경이 전혀 없습니다.

### 4.2 Injection 과정 (Action 예시)

```
Action Chunk: (chunk_size=16, action_dim=7) → flatten → (112,) vector
    ↓ Normalize to [-1, +1]
    ↓ Duplicate 12,544/112 = 112번 반복
    ↓ Reshape to (C'=16, H'=28, W'=28)
    ↓ Overwrite blank latent frame at action_latent_idx
Latent Frame에 action이 삽입됨
```

구체적 단계:
1. Action chunk `(K × d_act)` → flatten → `(K * d_act,)` 벡터
2. 각 action 차원을 `[-1, +1]`로 normalize
3. 벡터를 `ceil(C' * H' * W' / (K * d_act))` 번 반복 (duplicate)
4. `(C' × H' × W')` 크기로 reshape
5. 대상 latent frame을 이 값으로 overwrite

> 소스: `cosmos_policy/models/policy_text2world_model.py` → `replace_latent_with_action_chunk()`

### 4.3 Extraction 과정 (추론 시)

```
Generated Latent Frame at action_idx: (C'=16, H'=28, W'=28)
    ↓ Flatten → (12,544,)
    ↓ Reshape to (num_copies, K * d_act) → (112, 112) for LIBERO
    ↓ Average across all copies (dim=0)
    ↓ Reshape to (chunk_size=16, action_dim=7)
    ↓ Unnormalize from [-1, +1] to original scale
Final Action Chunk: (16, 7)
```

**플랫폼별 extraction 정밀도:**
- LIBERO: 12,544 / 112 = **112 copies** (나머지 0, 정확히 나눠짐)
- RoboCasa: 12,544 / 224 = **56 copies** (나머지 0, 정확히 나눠짐)
- ALOHA: 12,544 / 700 = **17 full copies** + 나머지 644 elements → **나머지는 무시**, 17개 full copy만 평균

> 소스: `cosmos_policy/experiments/robot/cosmos_utils.py` → `extract_action_chunk_from_latent_sequence()`

### 4.4 Future Proprio 예측 관련 참고사항

모델은 future proprio를 latent frame으로 예측하도록 학습되지만, 공개된 추론 헬퍼 API(`get_future_state_prediction()`)는 **future image predictions만 반환**합니다. Future proprio 예측값은 latent sequence 내에 존재하지만 사용자에게 직접 노출되지 않습니다.

### 4.5 Value Injection/Extraction
- Value는 **단일 스칼라** (expected return, [0, 1] 범위)
- Injection: normalize → `[-1, +1]` → 전체 `(C' × H' × W')` 볼륨에 동일 값으로 채움
- Extraction: latent frame의 모든 element 평균 → unnormalize → `[0, 1]`로 clamp

---

## 5. Latent Diffusion Sequence 구조

### 5.1 LIBERO (state_t = 9)

```
Index 0: [Blank]          ← VAE temporal compression용 placeholder
Index 1: [Current Proprio] ← 로봇 proprioception (dim=9), latent injection
Index 2: [Current Wrist]   ← 손목 카메라 이미지 (224×224), VAE encode
Index 3: [Current Primary]  ← 3인칭 카메라 이미지 (224×224), VAE encode
Index 4: [Action Chunk]     ← 16-step action (dim=7), latent injection
Index 5: [Future Proprio]   ← 미래 proprioception, latent injection
Index 6: [Future Wrist]     ← 미래 손목 이미지, VAE encode
Index 7: [Future Primary]   ← 미래 3인칭 이미지, VAE encode
Index 8: [Value]            ← V(s') 스칼라, latent injection
```

**Conditioning (clean frames)**: Index 0~3 (blank + proprio + wrist + primary)
**Target (noised → denoise)**: Index 4~8 (action + future state + value)

| 상수 | 값 |
|---|---|
| state_t | 9 |
| chunk_duration | 33 (1 + 8×4) |
| min/max_num_conditional_frames | 4 |
| ACTION_DIM | 7 |
| PROPRIO_DIM | 9 |
| NUM_ACTIONS_CHUNK | 16 |

### 5.2 RoboCasa (state_t = 11)

```
Index  0: [Blank]
Index  1: [Current Proprio]       ← dim=9
Index  2: [Current Wrist Image]
Index  3: [Current Primary Image]
Index  4: [Current Secondary Image]  ← 추가 3인칭 카메라
Index  5: [Action Chunk]           ← 32-step action (dim=7)
Index  6: [Future Proprio]
Index  7: [Future Wrist Image]
Index  8: [Future Primary Image]
Index  9: [Future Secondary Image]
Index 10: [Value]
```

| 상수 | 값 |
|---|---|
| state_t | 11 |
| chunk_duration | 41 (1 + 10×4) |
| min/max_num_conditional_frames | 5 |
| ACTION_DIM | 7 |
| PROPRIO_DIM | 9 |
| NUM_ACTIONS_CHUNK | 32 |

### 5.3 ALOHA (state_t = 11)

```
Index  0: [Blank]
Index  1: [Current Proprio]        ← dim=14 (14 joint angles)
Index  2: [Current Left Wrist]
Index  3: [Current Right Wrist]
Index  4: [Current Primary (Top-down)]
Index  5: [Action Chunk]            ← 50-step action (dim=14)
Index  6: [Future Proprio]
Index  7: [Future Left Wrist]
Index  8: [Future Right Wrist]
Index  9: [Future Primary]
Index 10: [Value]
```

| 상수 | 값 |
|---|---|
| state_t | 11 |
| chunk_duration | 41 (1 + 10×4) |
| min/max_num_conditional_frames | 5 |
| ACTION_DIM | 14 |
| PROPRIO_DIM | 14 |
| NUM_ACTIONS_CHUNK | 50 |

---

## 6. Training Pipeline

### 6.1 전체 학습 Flow

```
Robot Demonstration Data: (s_t, a_t, s_{t+K}, V(s_{t+K}))
    │
    ├─── Images (224×224×3) ──→ VAE Encode ──→ Latent Frames (16×28×28)
    ├─── Proprio (dim=9/14) ──→ Normalize [-1,+1] ──→ Latent Injection
    ├─── Action Chunk (K×d) ──→ Normalize [-1,+1] ──→ Latent Injection
    └─── Value (scalar) ──→ Normalize [-1,+1] ──→ Latent Injection
                │
                ▼
    Complete Latent Sequence: (B, C'=16, T'=state_t, H'=28, W'=28)
                │
    ┌───────────┴───────────┐
    │  Conditioning Frames  │  Target Frames (to denoise)
    │  (clean, σ≈0)        │  (noised with σ)
    │  [blank, proprio,     │  [action, future_proprio,
    │   wrist, primary]     │   future_wrist, future_primary,
    │                       │   value]
    └───────────┬───────────┘
                │
                ▼
    Add Noise: x_t = x_0 + σ·ε, where ε ~ N(0, I)
                │
                ▼
    DiT Forward Pass: D_θ(x_t; σ, c_text) → x̂_0
                │
                ▼
    Loss: ||x̂_0 - x_0||² × w(σ)    (EDM loss with per-sigma weights)
                │
                ▼
    Loss Masking (optional): frame별 selective loss
```

### 6.2 Balanced Batch Training (Joint Objectives)

#### Base Checkpoint 학습 (50/25/25)

초기 파인튜닝 단계에서 각 training batch는 **50/25/25**로 분할됩니다:

```
Training Batch (B samples) — Base Policy 학습
    │
    ├── 50% Demo Samples → Policy Training
    │   Conditioning: s (current state)
    │   Target: (a, s', V(s'))
    │   학습: π(a, s', V(s') | s)
    │
    ├── 25% Rollout Samples → World Model Training
    │   Conditioning: (s, a) — action frame도 clean으로 처리
    │   Target: (s', V(s'))
    │   학습: T̂(s', V(s') | s, a)
    │
    └── 25% Rollout Samples → Value Function Training
        Conditioning: (s, a, s') — 모든 frame clean
        Target: V(s') only
        학습: V(s' | s, a, s')
```

#### Planning Model 학습 (10/45/45) — Rollout Fine-Tuning

Planning을 위한 후속 파인튜닝 단계에서는 **배치 비율이 다릅니다**:

```
Training Batch (B samples) — Planning Model 학습
    │
    ├── 10% Demo Samples → Policy Training (최소한으로 유지)
    │
    ├── 45% Rollout Samples → World Model Training (강화)
    │
    └── 45% Rollout Samples → Value Function Training (강화)
```

> 논문 Section 4.3: "90 percent of each training batch is split evenly between training the world model and value function, while only 10 percent is used to train the policy."
> 소스: `config/experiment/cosmos_policy_experiment_configs.py:389` → `demonstration_sampling_prob=0.1`

### 6.3 Noise Distribution (Hybrid)

기존 EDM의 log-normal 분포를 수정하여 high-sigma 영역에 더 많은 가중치를 부여:

```
Training σ distribution:
  - 70% 확률: log-normal, ln(σ) ~ N(P_mean=1.39, P_std²=1.2²)
  - 30% 확률: uniform, σ ~ U(1.0, 85.0)

Training range: σ_min=0.01, σ_max=200
Inference range: σ_min=4, σ_max=80  ← 주의: inference에서 σ_min이 훨씬 높음
```

> 이유: 낮은 σ에서의 최종 denoising step이 오히려 action 정확도를 떨어뜨림

### 6.4 Training Hyperparameters

| 플랫폼 | GPUs | Batch Size | Steps | 시간 |
|---|---|---|---|---|
| LIBERO | 64 × H100 | 1920 | 40K | 48h |
| RoboCasa | 32 × H100 | 800 | 45K | 48h |
| ALOHA | 8 × H100 | 200 | 50K | 48h |

- Optimizer: AdamW, lr=1e-4
- LR Schedule: Cosine decay (LIBERO: 30K cycle, ALOHA: 20K cycle) → 5x decay
- Warmup: 1000-2000 steps
- Full fine-tuning (모든 model weight 업데이트)
- EMA: disabled

---

## 7. Inference Pipeline

### 7.1 Mode 1: Direct Policy (Without Planning)

```
┌─────────────────────────────────────────────────────────┐
│                    DIRECT POLICY MODE                    │
│              (Parallel Decoding, 빠름)                   │
└─────────────────────────────────────────────────────────┘

Step 1: 관측값 수집
  ├── Camera images (224×224×3) × N_cameras
  ├── Proprioception (dim=9 or 14)
  └── Task description → T5-XXL → text embeddings (512, 1024)

Step 2: 이미지 전처리
  ├── Resize to 224×224
  ├── Image augmentation (if trained with aug)
  └── 각 이미지 4장 복제 (temporal compression)

Step 3: VAE Encode
  ├── Image sequence → Wan2.1 VAE → Latent sequence
  │   (B, 3, 33, 224, 224) → (B, 16, 9, 28, 28) for LIBERO
  └── Proprioception → Latent Injection into frame

Step 4: Conditioning Setup
  ├── Conditional frames (Index 0~3): clean (σ ≈ 0)
  └── Target frames (Index 4~8): to be generated

Step 5: Diffusion Sampling (EDM)
  ├── Initialize x_T ~ N(0, σ_max²·I), σ_max = 80
  ├── For step in denoising_steps (5 or 10):
  │     ├── DiT forward: D_θ(x_t; σ_t, c_text) → x̂_0
  │     ├── Replace conditional frames with GT
  │     └── Update x_t via solver (2nd-order AB)
  └── Final: x̂_0 at σ_min = 4

Step 6: Action Extraction
  ├── x̂_0[:, :, action_idx, :, :] → (B, 16, 28, 28)
  ├── Flatten → reshape → average duplicates
  ├── Unnormalize from [-1,+1] to original scale
  └── Action chunk: (chunk_size, action_dim)

Step 7: Action Chunk 실행 (Suite별 상이)
  ├── LIBERO:   full chunk 실행 (16 steps 예측, 16 steps 실행)
  ├── RoboCasa: partial chunk 실행 (32 steps 예측, **앞 16 steps만 실행** 후 재질의)
  └── ALOHA:    full chunk 실행 (50 steps 예측, 50 steps 실행)
```

**Inference Latency (논문 Appendix A.4.2 기준):**
| 설정 | Denoising Steps | Latency (1×H100) |
|---|---|---|
| LIBERO/RoboCasa | 5 | 0.61 sec |
| ALOHA | 10 | 0.95 sec |
| 1-step (fast) | 1 | 0.16 sec |

**Open-Loop Execution 정책 (Suite별 상이):**
| Suite | 예측 chunk_size | 실행 num_open_loop_steps | 비고 |
|---|---|---|---|
| LIBERO | 16 | 16 (full chunk) | 전체 실행 후 재질의 |
| RoboCasa | 32 | **16 (half chunk)** | 앞 16 steps만 실행 후 재질의 |
| ALOHA | 50 | 50 (full chunk) | 전체 실행 후 재질의 |

> RoboCasa는 더 긴 action chunk를 예측하되 절반만 실행하여 receding-horizon 방식에 가깝게 동작합니다.
> 소스: `experiments/robot/robocasa/run_robocasa_eval.py:182-183`

### 7.2 Mode 2: Model-Based Planning (Best-of-N)

```
┌─────────────────────────────────────────────────────────┐
│              MODEL-BASED PLANNING MODE                   │
│         (Autoregressive Decoding, 정확하지만 느림)        │
└─────────────────────────────────────────────────────────┘

     ┌─────────────────────────────────────────┐
     │     Dual Deployment Architecture        │
     │                                         │
     │  Policy Model: 원본 Cosmos Policy 체크포인트  │
     │  Planning Model: rollout 데이터로 fine-tuned  │
     └─────────────────────────────────────────┘

Phase 1: Action Proposal Generation (Policy Model)
  ├── 현재 관측값 s 입력
  ├── N개의 action chunk 후보 샘플링 (N=8, 각 GPU 1개)
  ├── Denoising steps: 10
  └── 출력: {a₁, a₂, ..., a_N}  각각 (chunk_size, action_dim)

Phase 2: Future State Prediction (Planning Model, Autoregressive)
  ├── 각 action 후보 aᵢ에 대해:
  │   ├── **skip_vae_encoding=True**: VAE encoding 재수행 없이 Phase 1의 latent 재사용
  │   ├── **previous_generated_latent**: Phase 1에서 생성된 latent를 직접 전달
  │   ├── aᵢ를 conditioning으로 추가 (action frame → clean)
  │   ├── Future state frames만 denoising (5 steps)
  │   ├── 3회 반복 (ensemble)
  │   └── 출력: {ŝ'ᵢ,₁, ŝ'ᵢ,₂, ŝ'ᵢ,₃}
  └── 총 3×N = 24 future state predictions

Phase 3: Value Prediction (Planning Model, Autoregressive)
  ├── 각 future state prediction ŝ'ᵢ,ⱼ에 대해:
  │   ├── **skip_vae_encoding=True**: Phase 2의 latent 재사용
  │   ├── **previous_generated_latent**: future state latent를 직접 전달
  │   ├── (s, a, ŝ') 또는 (ŝ')만을 conditioning으로 사용
  │   │   - V(s') 모드: current state + action 마스킹
  │   │   - Q(s,a) 모드: future state 마스킹
  │   ├── Value frame만 denoising (5 steps)
  │   ├── 5회 반복 (ensemble)
  │   └── 출력: {V̂ᵢ,ⱼ,₁, ..., V̂ᵢ,ⱼ,₅}
  └── 총 5×3×N = 120 value predictions

Phase 4: Action Selection
  ├── 각 action 후보별 15개 value predictions 집계
  ├── "Majority Mean" 방식:
  │   ├── 과반수가 success/failure 판단 (threshold 기준)
  │   └── 과반수 그룹의 평균 value 계산
  ├── 가장 높은 value의 action 선택
  └── 선택된 action chunk 전체 실행

총 Latency: ~4.9 sec (8×H100 GPU 병렬)
```

### 7.3 Search Depth > 1 (Multi-Step Lookahead)

코드는 `search_depth > 1`을 지원하여 더 깊은 탐색 트리를 구성할 수 있습니다:

```
search_depth=1 (기본): s → a → s' → V(s')
search_depth=2:        s → a₁ → s'₁ → a₂ → s'₂ → V(s'₂)
search_depth=K:        s → a₁ → s'₁ → ... → aₖ → s'ₖ → V(s'ₖ)
```

구현 방식:
1. Phase 1-3을 통해 action, future state, value 예측
2. 예측된 future state의 latent frames를 current state 자리에 교체
3. 교체된 latent를 입력으로 다시 action → future state → value 예측 반복
4. 전체 depth에 걸친 value를 종합하여 최종 action 선택

> 소스: `cosmos_policy/experiments/robot/cosmos_utils.py:2013-2080`

### 7.4 Value Ensemble 집계 방식

Value prediction ensemble의 집계에는 4가지 방식이 지원됩니다:

| 방식 | 설명 |
|---|---|
| `average` | 단순 평균 |
| `lcb` | Lower Confidence Bound (평균 - α × 표준편차) |
| `success_vote` | 성공/실패 threshold 기반 다수결 투표 |
| `majority_mean` | 과반수 그룹 결정 후 해당 그룹 내 평균 (논문 기본) |

### 7.5 Autoregressive Phase에서의 Latent 재사용 최적화

Planning의 Phase 2, 3에서는 VAE encoding을 재수행하지 않습니다:
- `skip_vae_encoding=True`: 이전 phase에서 생성된 latent를 직접 재사용
- `previous_generated_latent`: 이전 denoising 결과를 다음 phase의 입력으로 전달
- 이로 인해 각 phase는 conditioning mask만 변경하고 새로운 target slot만 denoising

> 소스: `cosmos_policy/experiments/robot/cosmos_utils.py:1343-1344, 1492-1493`

---

## 8. DiT Forward Pass 상세

### 8.1 Input Preparation

```
x_t: (B, C'=16, T'=state_t, H'=28, W'=28)
  ↓ Concatenate condition_video_input_mask (B, 1, T', H', W')  [MinimalV1LVGDiT]
  ↓ Concatenate padding_mask (B, 1, T', H', W')                [MiniTrainDIT, concat_padding_mask=True]
  ↓ 결과: (B, 18, T', 28, 28)  ← 실제 Patch Embed 입력
  ↓ Patchify (patch_spatial=2, patch_temporal=1)
  ↓ 각 patch: (18, 1, 2, 2) → flatten → (18×1×2×2 = 72)
  ↓ Linear projection → (model_channels = 2048)

Sequence length = T' × (H'/patch_s) × (W'/patch_s)
  LIBERO: 9 × 14 × 14 = 1,764 tokens
  ALOHA:  11 × 14 × 14 = 2,156 tokens

※ condition_video_input_mask: 어떤 frame이 clean conditioning인지 표시 (1=clean, 0=noised)
※ padding_mask: 이미지 패딩 영역 표시
※ 출력은 여전히 16채널 (추가 2채널은 입력 conditioning 신호일 뿐)
```

### 8.2 Transformer Block (×28)

```
Input: x ∈ R^(B, S, 2048)    where S = sequence length
  │
  ├── RMSNorm
  ├── Self-Attention (Multi-Head)
  │   ├── Q, K, V = Linear(2048 → 2048) each
  │   ├── Reshape: (B, S, 16, 128) → Q, K에 RoPE3D 적용
  │   ├── RMSNorm on Q, K
  │   ├── Scaled Dot-Product Attention
  │   └── Output Projection: Linear(2048 → 2048)
  │
  ├── + Residual Connection
  │
  ├── RMSNorm
  ├── Cross-Attention (Text Conditioning)
  │   ├── Q from x: Linear(2048 → 2048)
  │   ├── K, V from text_emb: Linear(1024 → 2048) each
  │   ├── Attention with text context (max 512 tokens)
  │   └── Output Projection: Linear(2048 → 2048)
  │
  ├── + Residual Connection
  │
  ├── RMSNorm
  ├── MLP (Feed-Forward)
  │   ├── Linear(2048 → 8192) + GELU
  │   └── Linear(8192 → 2048)
  │
  └── + Residual Connection

※ AdaLN (Adaptive Layer Norm): σ 값을 조건으로 scale/shift 파라미터 조정
  σ → Timestep Embedding → AdaLN-LoRA(dim=256) → scale, shift for each LayerNorm
```

### 8.3 Output

```
Transformer Output: (B, S, 2048)
  ↓ Unpatchify: reverse of patchification
  ↓ Linear projection → (C'=16, 1, 2, 2) per patch
  ↓ Reshape back to (B, C'=16, T', H'=28, W'=28)

EDM Preconditioning:
  x̂_0 = c_skip(σ) · x_t + c_out(σ) · net_output
  ε̂ = (x_t - x̂_0) / σ
```

---

## 9. Conditioning Mask 메커니즘

### 9.1 Video Conditioning (Frame Replace)

```
condition_video_input_mask: (B, 1, T', H', W')
  - 1 = clean conditioning frame (σ_conditional ≈ 0)
  - 0 = noised target frame (σ from noise schedule)

LIBERO 기본: [1, 1, 1, 1, 0, 0, 0, 0, 0]
              blank proprio wrist primary action f_proprio f_wrist f_primary value
```

### 9.2 Training Mode별 Mask 변경

```
Policy Training (Demo Data):
  Mask: [1, 1, 1, 1, 0, 0, 0, 0, 0]  ← Index 0~3 clean, 4~8 noised
  Loss:                  ↑action   ↑future_state   ↑value (auxiliary)

World Model Training (Rollout Data):
  Mask: [1, 1, 1, 1, 1, 0, 0, 0, 0]  ← Index 4(action)도 clean으로!
  Loss:                     ↑future_state   ↑value (auxiliary)

Value Function Training (Rollout Data):
  Mask: [1, 1, 1, 1, 1, 1, 1, 1, 0]  ← value만 noised
  Loss:                              ↑value only
```

### 9.3 Value Function Variants (Planning 시)

```
V(s') 모드: current state + action 마스킹
  Input Mask: [0, 0, 0, 0, 0, 1, 1, 1, 0]
  → 미래 상태만으로 value 예측

Q(s, a) 모드: future state 마스킹
  Input Mask: [1, 1, 1, 1, 1, 0, 0, 0, 0]
  → 현재 상태 + action으로 value 예측 (model-free)
```

---

## 10. Diffusion Sampling (EDM Framework)

### 10.1 SDE 파라미터 (Inference)

```python
# HybridEDMSDE 설정
sigma_max = 80        # Training: 200
sigma_min = 4         # Training: 0.01
sigma_data = 0.5      # EDM default
```

### 10.2 EDM Preconditioning

```
c_skip(σ) = σ_data² / (σ² + σ_data²)
c_out(σ) = σ · σ_data / √(σ² + σ_data²)
c_in(σ) = 1 / √(σ² + σ_data²)
c_noise(σ) = ln(σ) / 4
```

### 10.3 Sampling Schedule

```
Denoising Steps (LIBERO/RoboCasa): 5 steps
Denoising Steps (ALOHA):           10 steps
Solver: 2nd-order Adams-Bashforth ("2ab")

σ schedule: Karras/EDM 스타일 timestamp 생성 (rho=7)
  - get_rev_ts(t_min=σ_min, t_max=σ_max, num_steps, order=rho) 사용
  - rho 파라미터가 high/low sigma 간 spacing 분포를 제어
  - geometric spacing이 아닌 rho-weighted spacing
  - 높은 rho 값(7)은 낮은 σ 근처에 더 많은 step을 배치

※ σ의 정확한 값은 rho, num_steps에 따라 달라지므로 고정 리스트로 표현 불가
```

> 소스: `cosmos_policy/modules/cosmos_sampler.py:60,90,121-122`

### 10.4 Conditional Frame Handling during Denoising

```
Each denoising step:
  1. net_input = x_t * c_in(σ)
  2. For conditional frames (mask=1):
     net_input[:, :, cond_idx] = gt_frames / σ_data  (clean input)
     c_noise for cond_idx = ln(σ_conditional) / 4     (near-zero noise)
  3. DiT forward → net_output
  4. x̂_0 = c_skip · x_t + c_out · net_output
  5. For conditional frames:
     x̂_0[:, :, cond_idx] = gt_frames  (replace with GT)
  6. Update x_t via ODE solver
```

---

## 11. Dimension Reference Table

### 11.1 Latent Space Dimensions

| 항목 | 값 |
|---|---|
| Latent Channels (C') | 16 |
| Latent Height (H') | 28 (= 224/8) |
| Latent Width (W') | 28 (= 224/8) |
| Elements per Latent Frame | 12,544 (= 16×28×28) |
| Image Resolution | 224 × 224 |

### 11.2 Robot Platform별 Dimensions

| | LIBERO | RoboCasa | ALOHA |
|---|---|---|---|
| state_t (latent frames) | 9 | 11 | 11 |
| chunk_duration (images) | 33 | 41 | 41 |
| conditional_frames | 4 | 5 | 5 |
| ACTION_DIM | 7 | 7 | 14 |
| PROPRIO_DIM | 9 | 9 | 14 |
| NUM_ACTIONS_CHUNK | 16 | 32 | 50 |
| Action elements | 112 | 224 | 700 |
| Action copies in latent | 112 (정확히 나눠짐) | 56 (정확히 나눠짐) | 17 (나머지 644 무시) |
| Camera views | 2 (wrist+primary) | 3 (wrist+2×third) | 3 (2×wrist+primary) |
| Controller frequency | - (미명시) | - (미명시) | 25 Hz (논문 Section 5.1) |
| Discount factor (γ) | 0.99 | 0.99 | 0.998 |

### 11.3 DiT Token Dimensions

| 항목 | LIBERO | ALOHA |
|---|---|---|
| Temporal tokens | 9 | 11 |
| Spatial tokens per frame | 14 × 14 = 196 | 14 × 14 = 196 |
| Total sequence length | 1,764 | 2,156 |
| Hidden dim per token | 2,048 | 2,048 |

---

## 12. 코드 파일 구조 매핑

### 12.1 Core Model Files

| 파일 | 역할 |
|---|---|
| `models/policy_video2world_model.py` | 메인 모델 클래스 (CosmosPolicyVideo2WorldModel) |
| `models/policy_text2world_model.py` | 기반 Policy Diffusion Model + latent injection 함수 |
| `_src/predict2/networks/minimal_v1_lvg_dit.py` | MinimalV1LVGDiT (condition mask 채널 추가 wrapper) |
| `_src/predict2/networks/minimal_v4_dit.py` | MiniTrainDIT 기반 클래스 (패치 임베딩, 트랜스포머 블록 등) |
| `_src/predict2/models/text2world_model.py` | Base Diffusion Model |
| `_src/predict2/models/video2world_model.py` | Video2World 베이스 |

### 12.2 Tokenizer & Network

| 파일 | 역할 |
|---|---|
| `tokenizers/wan2pt1.py` | Wan2pt1VAEInterface — VAE 인코더/디코더 인터페이스 (Cosmos Policy용) |
| `_src/predict2/tokenizers/wan2pt1.py` | Wan2pt1VAEInterface — VAE 인코더/디코더 (base predict2) |
| `_src/predict2/networks/wan2pt1.py` | WanModel — Wan2.1 **diffusion backbone** (VAE가 아님, 별도 네트워크) |
| `config/defaults/tokenizer.py` | Tokenizer 설정 등록 |

### 12.3 Training & Inference

| 파일 | 역할 |
|---|---|
| `config/experiment/cosmos_policy_experiment_configs.py` | 전체 실험 설정 |
| `modules/hybrid_edm_sde.py` | Hybrid noise distribution (log-normal + uniform) |
| `modules/cosmos_sampler.py` | Diffusion sampling |
| `experiments/robot/cosmos_utils.py` | 추론 유틸리티 (get_action, extract_action 등) |
| `experiments/robot/aloha/deploy.py` | ALOHA 실시간 배포 서버 |
| `constants.py` | 플랫폼별 상수 (ACTION_DIM, PROPRIO_DIM 등) |

### 12.4 Class Hierarchy

```
ImaginaireModel
  └── DiffusionModel (Text2WorldModel)
        └── CosmosPolicyDiffusionModel (policy_text2world_model.py)
              └── CosmosPolicyVideo2WorldModel (policy_video2world_model.py)
                    ├── Video conditioning (gt_frames)
                    ├── Conditioning mask manipulation
                    ├── High sigma strategies
                    └── Latent injection for actions/proprio/value
```

---

## 13. End-to-End Inference Flow Diagram (Text)

```
                        ┌─────────────────┐
                        │  Robot Sensors   │
                        │  ├── Camera ×N   │
                        │  └── Joint Angles│
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
              ┌─────▼─────┐           ┌──────▼──────┐
              │  Images    │           │   Proprio   │
              │ 224×224×3  │           │  dim=9/14   │
              │  × N_cams  │           │             │
              └─────┬─────┘           └──────┬──────┘
                    │                         │
              ┌─────▼─────┐           ┌──────▼──────┐
              │ 4× 복제    │           │  Normalize  │
              │ per image  │           │  [-1, +1]   │
              └─────┬─────┘           └──────┬──────┘
                    │                         │
              ┌─────▼─────────┐       ┌──────▼──────┐
              │  Wan2.1 VAE   │       │   Latent    │
              │   Encoder     │       │  Injection  │
              │ → (16,28,28)  │       │ → (16,28,28)│
              └─────┬─────────┘       └──────┬──────┘
                    │                         │
                    └────────────┬────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Latent Sequence      │
                    │  (B, 16, T', 28, 28)  │
                    │  T'=9 (LIBERO)        │
                    │  T'=11 (ALOHA)        │
                    └───────────┬───────────┘
                                │
               ┌────────────────┴────────────────┐
               │  Conditioning Mask Setup         │
               │  ├── Cond frames: clean (σ≈0)   │
               │  └── Target frames: noised      │
               └────────────────┬────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Initialize x_T       │
                    │  ~ N(0, σ_max²·I)     │
                    │  σ_max = 80           │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Denoising Loop       │
                    │  (5 or 10 steps)      │◄──── T5 Text Embeddings
                    │                       │      (512, 1024)
                    │  ┌─────────────────┐  │
                    │  │ DiT (2B params) │  │
                    │  │ 28 blocks       │  │
                    │  │ 16 heads        │  │
                    │  │ dim=2048        │  │
                    │  └─────────────────┘  │
                    │                       │
                    │  x̂_0 = c_skip·x_t    │
                    │       + c_out·output  │
                    │  Replace cond frames  │
                    │  with GT              │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Generated Latent     │
                    │  x̂_0 at σ_min=4      │
                    │  (B, 16, T', 28, 28)  │
                    └───────────┬───────────┘
                                │
               ┌────────────────┼────────────────┐
               │                │                │
        ┌──────▼──────┐ ┌──────▼──────┐ ┌───────▼───────┐
        │Action Frame │ │Future State │ │  Value Frame  │
        │  Extract    │ │  VAE Decode │ │   Extract     │
        │  Average    │ │  → Images   │ │   Average     │
        │  Unnorm     │ │             │ │   Unnorm      │
        └──────┬──────┘ └──────┬──────┘ └───────┬───────┘
               │                │                │
        ┌──────▼──────┐ ┌──────▼──────┐ ┌───────▼───────┐
        │ Action Chunk│ │Future Images│ │  V(s') ∈[0,1] │
        │ (K, d_act)  │ │ 224×224×3   │ │  scalar       │
        └──────┬──────┘ └─────────────┘ └───────────────┘
               │
        ┌──────▼──────┐
        │  Execute on │
        │    Robot    │
        └─────────────┘
```

---

## 14. Planning Flow Diagram (Best-of-N)

```
                    ┌─────────────────────┐
                    │   Current State s   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   POLICY MODEL      │
                    │   (원본 체크포인트)    │
                    │                     │
                    │  N=8 action 샘플링   │
                    │  10 denoising steps  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  {a₁, a₂, ..., a₈}  │
                    │  각 (chunk, d_act)   │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼──── × 8 (parallel GPUs) ───┐
              │                │                             │
    ┌─────────▼─────────┐                        ┌──────────▼──────────┐
    │  PLANNING MODEL    │          ...           │   PLANNING MODEL    │
    │  (rollout FT)      │                        │   (rollout FT)      │
    │                    │                        │                     │
    │  Future State ×3   │                        │   Future State ×3   │
    │  (5 steps each)    │                        │   (5 steps each)    │
    └─────────┬─────────┘                        └──────────┬──────────┘
              │                                              │
    ┌─────────▼─────────┐                        ┌──────────▼──────────┐
    │  Value ×5 per s'   │          ...           │   Value ×5 per s'   │
    │  (5 steps each)    │                        │   (5 steps each)    │
    │  Total: 15 values  │                        │   Total: 15 values  │
    └─────────┬─────────┘                        └──────────┬──────────┘
              │                                              │
              └────────────────┬─────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Majority Mean     │
                    │   Aggregation       │
                    │                     │
                    │  각 aᵢ → 15 values  │
                    │  → majority vote    │
                    │  → group mean       │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Best Action = argmax│
                    │  Execute on Robot   │
                    └─────────────────────┘

총 Inference per step:
  - 8 action proposals × 10 denoising steps
  - 24 future state predictions × 5 denoising steps
  - 120 value predictions × 5 denoising steps
  - Total wall time: ~4.9 sec on 8×H100
```

---

## 15. Key Design Decisions 요약

1. **No architecture modification**: 기존 video diffusion model의 구조를 전혀 변경하지 않음
2. **Latent Frame Injection**: 새로운 modality를 latent frame으로 인코딩하여 직접 삽입
3. **Single-stage fine-tuning**: 별도의 action module 없이 한 단계의 fine-tuning으로 policy 학습
4. **Unified model**: 하나의 모델이 policy + world model + value function 역할 동시 수행
5. **Hybrid noise distribution**: Action 정확도 향상을 위해 high-sigma 영역에 더 많은 training weight
6. **Elevated σ_min at inference**: σ_min=4로 설정하여 low-noise regime의 부정확한 denoising 방지
7. **Auxiliary supervision**: Policy 학습 시 future state와 value도 함께 예측하여 성능 향상 (+1.5%)
8. **Pretrained backbone**: Video model의 spatiotemporal prior가 action prediction에도 유효 (+3.9%)
