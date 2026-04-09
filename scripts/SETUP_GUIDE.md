# New Server Setup Guide

## Overview

이 가이드는 Cosmos-Policy 실험 환경을 새 서버에 구축하는 절차를 설명합니다.
기존 서버에서 발견된 모든 시행착오(환경 충돌, 라이브러리 심링크, robosuite 로그 등)가 자동화 스크립트에 반영되어 있습니다.

## Prerequisites

```bash
# 1. CUDA 12.x 드라이버 확인
nvidia-smi

# 2. uv 설치 (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. cmake 설치 (egl-probe 빌드에 필요)
sudo apt install cmake

# 4. gh CLI 설치 (GitHub 인증)
# https://cli.github.com/

# 5. HuggingFace 로그인 (모델 다운로드)
pip install huggingface-hub
huggingface-cli login
```

## Quick Start (5분)

```bash
# 1. Repos 클론
git clone https://github.com/JHLEE17/cosmos-policy.git  # fork
git clone https://github.com/JHLEE17/adaptive-ttc-wam.git
cd cosmos-policy

# 2. 자동 셋업 (venv 생성 + 의존성 + 심링크 + 모델 다운로드)
bash scripts/setup_new_server.sh --with-libero

# 3. 설치 확인
.venv/bin/python3 -c "import cosmos_policy; print('OK')"

# 4. 실험 실행 (Phase A, GPU 0 단독)
bash scripts/run_experiment.sh --phase a --gpu 0

# 또는 멀티 GPU 병렬
bash scripts/run_experiment.sh --phase a --gpu 0,1,2,3 --parallel
```

## Detailed Steps

### Step 1: Repository Setup

```bash
# cosmos-policy fork (분석 코드 포함)
git clone https://github.com/JHLEE17/cosmos-policy.git
cd cosmos-policy

# adaptive-ttc-wam (실험 결과 저장소)
cd ..
git clone https://github.com/JHLEE17/adaptive-ttc-wam.git
cd cosmos-policy
```

### Step 2: Environment Setup

```bash
# 자동 스크립트 사용 (권장)
bash scripts/setup_new_server.sh --with-libero

# 수동 설치 시:
uv venv --python 3.10 .venv
uv sync --group libero
# + nvidia 심링크, sitecustomize.py 등은 스크립트 참고
```

### Step 3: GPU 할당 및 실험 실행

B200 서버에서 멀티 GPU 병렬 실행:

```bash
# Phase A: LIBERO-Spatial 5-step denoising (10 tasks)
# 4 GPU 병렬 → tasks를 자동 분배 (GPU당 ~2-3 tasks)
bash scripts/run_experiment.sh --phase a --gpu 0,1,2,3 --parallel

# 특정 task 범위만 실행
bash scripts/run_experiment.sh --phase a --gpu 0 --task-start 0 --task-end 3

# 직접 Python 실행 (fine-grained control)
SHIM_DIR=".venv/lib/nvidia-libs"
NVIDIA_LIBS=$(find .venv/lib/python3.10/site-packages/nvidia -name lib -type d | paste -sd:)
export PATH="$SHIM_DIR:$PATH"
export LD_LIBRARY_PATH="$SHIM_DIR:$NVIDIA_LIBS"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 \
    -m cosmos_policy.experiments.robot.analysis.collect_candidate_data \
    --config cosmos_predict2_2b_480p_libero__inference_only \
    --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
    --config_file cosmos_policy/config/config.py \
    --use_wrist_image True --use_proprio True \
    --normalize_proprio True --unnormalize_actions True \
    --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
    --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
    --trained_with_image_aug True \
    --chunk_size 16 --num_open_loop_steps 16 \
    --task_suite_name libero_spatial \
    --num_trials_per_task 10 --seed 195 --deterministic True \
    --use_jpeg_compression True --flip_images True \
    --num_denoising_steps_action 1 \
    --num_denoising_steps_future_state 5 \
    --num_denoising_steps_value 5 \
    --num_candidates 8 \
    --num_wm_rollouts_full 3 --num_value_preds_full 5 \
    --output_dir ../adaptive-ttc-wam/experiments/0Zero_value_validity/phase_a_5step \
    --task_start 0 --task_end 5
```

### Step 4: 결과 병합 및 분석

```bash
PYTHON=".venv/bin/python3"
OUTPUT_DIR="../adaptive-ttc-wam/experiments/0Zero_value_validity/phase_a_5step"
ANALYSIS_DIR="${OUTPUT_DIR}_results"

# 병합
$PYTHON -m cosmos_policy.experiments.robot.analysis.merge_all_tasks \
    --output_dir "$OUTPUT_DIR" --suite libero_spatial

# 분석 (4개)
MERGED="$OUTPUT_DIR/candidate_data_libero_spatial_merged.json"
$PYTHON -m cosmos_policy.experiments.robot.analysis.value_stability_analysis \
    --input_path "$MERGED" --output_dir "$ANALYSIS_DIR"
$PYTHON -m cosmos_policy.experiments.robot.analysis.value_accuracy_analysis \
    --input_path "$MERGED" --output_dir "$ANALYSIS_DIR"
$PYTHON -m cosmos_policy.experiments.robot.analysis.rank_correlation_analysis \
    --input_path "$MERGED" --output_dir "$ANALYSIS_DIR"
$PYTHON -m cosmos_policy.experiments.robot.analysis.uncertainty_reversal_analysis \
    --input_path "$MERGED" --output_dir "$ANALYSIS_DIR"
```

## Known Issues & Fixes

이 스크립트들은 아래 문제들을 자동으로 처리합니다:

| 문제 | 원인 | 스크립트의 해결 |
|------|------|-----------------|
| `libcudnn.so not found` | transformer_engine이 unversioned .so 로드 | nvidia-libs/ 에 symlink 생성 |
| `ldconfig -p` 에서 libnvrtc 못 찾음 | LD_LIBRARY_PATH 무시 | fake ldconfig script in PATH |
| LIBERO interactive prompt | `~/.libero/config.yaml` 없음 | 자동 생성 |
| `/tmp/robosuite.log` permission denied | 다른 유저 소유 파일 | sitecustomize.py monkey-patch |

## GPU Memory 참고

| 구성 | VRAM 사용 | 비고 |
|------|-----------|------|
| 프로세스 1개 | ~6.5 GB | 기본 |
| 프로세스 2개 (같은 GPU) | ~13 GB | 병렬 가능 확인됨 |
| B200 (192GB) | - | GPU당 10+ 프로세스 이론적 가능, 실제 2-3개 권장 |

## Experiment Matrix

| Phase | Benchmark | Denoising | 목적 |
|-------|-----------|-----------|------|
| A | LIBERO-Spatial | action=1, fs=5, v=5 | Value noise가 denoising 탓인지 확인 |
| B | RoboCasa | TBD | Harder benchmark에서 value validity 재검증 |

## File Structure

```
cosmos-policy/
  scripts/
    setup_new_server.sh     # 환경 자동 구축
    run_experiment.sh        # 실험 실행 (Phase A/B)
    SETUP_GUIDE.md           # 이 문서
  cosmos_policy/experiments/robot/analysis/
    collect_candidate_data.py    # 데이터 수집 (GPU)
    merge_all_tasks.py           # JSON shard 병합
    value_stability_analysis.py  # 0-Zero-A: ICC, variance decomposition
    value_accuracy_analysis.py   # 0-Zero-B: value-accuracy correlation
    rank_correlation_analysis.py # 0-A: shallow-full rank correlation
    uncertainty_reversal_analysis.py # 0-B: uncertainty-reversal prediction

adaptive-ttc-wam/
  experiments/
    0Zero_value_validity/    # Phase A 결과
    0A_rank_correlation/     # Exp 0-A 결과
    0B_uncertainty_reversal/ # Exp 0-B 결과
  notes/
    Research_Plan_draft.md   # 전체 연구 계획
    exp0zero_phase_a_progress.md  # Phase A 진행 문서
```
