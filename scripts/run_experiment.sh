#!/bin/bash
# ============================================================
# Run Adaptive TTC-WAM Experiments
# ============================================================
# Usage:
#   bash scripts/run_experiment.sh --phase a --gpu 0 [--task-start 0 --task-end 5]
#   bash scripts/run_experiment.sh --phase a --gpu 0,1 --parallel
#   bash scripts/run_experiment.sh --phase b --gpu 0 --suite robocasa
#
# Phases:
#   a: 5-step denoising value validity (LIBERO-Spatial)
#   b: RoboCasa baseline + planning pilot
# ============================================================

set -euo pipefail

# ==========================================
# Defaults
# ==========================================
PHASE="a"
GPU="0"
PARALLEL=false
TASK_START=""
TASK_END=""
SUITE="libero_spatial"
NUM_TRIALS=10
OUTPUT_BASE=""

# ==========================================
# Parse arguments
# ==========================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --phase) PHASE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --parallel) PARALLEL=true; shift ;;
        --task-start) TASK_START="$2"; shift 2 ;;
        --task-end) TASK_END="$2"; shift 2 ;;
        --suite) SUITE="$2"; shift 2 ;;
        --num-trials) NUM_TRIALS="$2"; shift 2 ;;
        --output-dir) OUTPUT_BASE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# ==========================================
# Environment setup
# ==========================================
PYTHON="$REPO_ROOT/.venv/bin/python3"
NVIDIA_BASE="$REPO_ROOT/.venv/lib/python3.10/site-packages/nvidia"
SHIM_DIR="$REPO_ROOT/.venv/lib/nvidia-libs"
ALL_NVIDIA_LIBS=$(find "$NVIDIA_BASE" -name "lib" -type d 2>/dev/null | paste -sd: || echo "")

export PATH="$SHIM_DIR:$PATH"
export LD_LIBRARY_PATH="$SHIM_DIR:${ALL_NVIDIA_LIBS:-}"

# ==========================================
# Phase A: 5-step denoising on LIBERO-Spatial
# ==========================================
if [ "$PHASE" = "a" ]; then
    [ -z "$OUTPUT_BASE" ] && OUTPUT_BASE="$REPO_ROOT/../adaptive-ttc-wam/experiments/0Zero_value_validity/phase_a_5step"
    mkdir -p "$OUTPUT_BASE"

    COMMON_ARGS=(
        --config cosmos_predict2_2b_480p_libero__inference_only
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B
        --config_file cosmos_policy/config/config.py
        --use_wrist_image True
        --use_proprio True
        --normalize_proprio True
        --unnormalize_actions True
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl
        --trained_with_image_aug True
        --chunk_size 16
        --num_open_loop_steps 16
        --task_suite_name "$SUITE"
        --num_trials_per_task "$NUM_TRIALS"
        --seed 195
        --deterministic True
        --use_jpeg_compression True
        --flip_images True
        --num_denoising_steps_action 1
        --num_denoising_steps_future_state 5
        --num_denoising_steps_value 5
        --num_candidates 8
        --num_wm_rollouts_full 3
        --num_value_preds_full 5
        --output_dir "$OUTPUT_BASE"
    )

    if $PARALLEL; then
        # Split tasks across GPUs
        IFS=',' read -ra GPUS <<< "$GPU"
        NUM_GPUS=${#GPUS[@]}
        TOTAL_TASKS=10
        TASKS_PER_GPU=$(( (TOTAL_TASKS + NUM_GPUS - 1) / NUM_GPUS ))

        echo "============================================"
        echo "Phase A: 5-step Denoising (PARALLEL)"
        echo "  GPUs: ${GPUS[*]} ($NUM_GPUS GPUs)"
        echo "  Tasks per GPU: ~$TASKS_PER_GPU"
        echo "  Date: $(date)"
        echo "============================================"

        PIDS=()
        for i in "${!GPUS[@]}"; do
            START=$(( i * TASKS_PER_GPU ))
            END=$(( (i + 1) * TASKS_PER_GPU ))
            [ $END -gt $TOTAL_TASKS ] && END=$TOTAL_TASKS
            [ $START -ge $TOTAL_TASKS ] && continue

            GPU_ID=${GPUS[$i]}
            LOG="$OUTPUT_BASE/gpu${GPU_ID}_tasks${START}-$((END-1)).log"

            echo "[GPU $GPU_ID] Starting tasks $START-$((END-1))..."
            CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON \
                -m cosmos_policy.experiments.robot.analysis.collect_candidate_data \
                "${COMMON_ARGS[@]}" \
                --task_start $START --task_end $END \
                > "$LOG" 2>&1 &
            PIDS+=($!)
            echo "  PID: ${PIDS[-1]}, Log: $LOG"
        done

        echo ""
        echo "Waiting for all GPUs to finish..."
        for pid in "${PIDS[@]}"; do
            wait $pid || echo "WARNING: PID $pid exited with error"
        done
        echo "All batches complete!"

    else
        # Single GPU
        [ -z "$TASK_START" ] && TASK_START=0
        [ -z "$TASK_END" ] && TASK_END=10

        echo "============================================"
        echo "Phase A: 5-step Denoising (GPU $GPU)"
        echo "  Tasks: $TASK_START to $((TASK_END-1))"
        echo "  Date: $(date)"
        echo "============================================"

        CUDA_VISIBLE_DEVICES=$GPU $PYTHON \
            -m cosmos_policy.experiments.robot.analysis.collect_candidate_data \
            "${COMMON_ARGS[@]}" \
            --task_start "$TASK_START" --task_end "$TASK_END" \
            2>&1 | tee "$OUTPUT_BASE/gpu${GPU}_tasks${TASK_START}-$((TASK_END-1)).log"
    fi

    # Merge shards
    echo ""
    echo "Merging results..."
    $PYTHON -m cosmos_policy.experiments.robot.analysis.merge_all_tasks \
        --output_dir "$OUTPUT_BASE" --suite "$SUITE"

    # Run analyses
    MERGED_FILE="$OUTPUT_BASE/candidate_data_${SUITE}_merged.json"
    ANALYSIS_DIR="${OUTPUT_BASE}_results"
    mkdir -p "$ANALYSIS_DIR"

    echo "Running analyses..."
    $PYTHON -m cosmos_policy.experiments.robot.analysis.value_stability_analysis \
        --input_path "$MERGED_FILE" --output_dir "$ANALYSIS_DIR"
    $PYTHON -m cosmos_policy.experiments.robot.analysis.value_accuracy_analysis \
        --input_path "$MERGED_FILE" --output_dir "$ANALYSIS_DIR"
    $PYTHON -m cosmos_policy.experiments.robot.analysis.rank_correlation_analysis \
        --input_path "$MERGED_FILE" --output_dir "$ANALYSIS_DIR"
    $PYTHON -m cosmos_policy.experiments.robot.analysis.uncertainty_reversal_analysis \
        --input_path "$MERGED_FILE" --output_dir "$ANALYSIS_DIR"

    echo ""
    echo "============================================"
    echo "Phase A Complete! Results: $ANALYSIS_DIR"
    echo "============================================"

# ==========================================
# Phase B: RoboCasa baseline + pilot
# ==========================================
elif [ "$PHASE" = "b" ]; then
    [ -z "$OUTPUT_BASE" ] && OUTPUT_BASE="$REPO_ROOT/../adaptive-ttc-wam/experiments/phase_b_robocasa"
    mkdir -p "$OUTPUT_BASE"

    echo "============================================"
    echo "Phase B: RoboCasa Baseline (GPU $GPU)"
    echo "  Suite: $SUITE"
    echo "  Date: $(date)"
    echo "============================================"

    # TODO: RoboCasa config and checkpoint paths
    echo "Phase B implementation pending."
    echo "Requires: RoboCasa checkpoint, task definitions"

else
    echo "Unknown phase: $PHASE (use 'a' or 'b')"
    exit 1
fi
