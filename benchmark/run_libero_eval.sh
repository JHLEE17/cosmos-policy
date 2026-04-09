#!/bin/bash
set -e
cd /workspace

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")
LOG_DIR="benchmark/libero_results"
mkdir -p "$LOG_DIR"

# Start GPU monitoring
bash benchmark/monitor_gpu.sh "$LOG_DIR/gpu_monitor.csv" &
GPU_MONITOR_PID=$!

echo "=== LIBERO Evaluation Started: $(date) ===" | tee "$LOG_DIR/eval_summary.log"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee -a "$LOG_DIR/eval_summary.log"

for suite in "${SUITES[@]}"; do
    echo "" | tee -a "$LOG_DIR/eval_summary.log"
    echo "=== Starting $suite: $(date) ===" | tee -a "$LOG_DIR/eval_summary.log"
    START_TIME=$(date +%s)
    
    uv run --extra cu128 --group libero --python 3.10 \
      python -m cosmos_policy.experiments.robot.libero.run_libero_eval \
        --config cosmos_predict2_2b_480p_libero__inference_only \
        --ckpt_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-LIBERO-Predict2-2B/libero_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 16 \
        --num_open_loop_steps 16 \
        --task_suite_name "$suite" \
        --local_log_dir cosmos_policy/experiments/robot/libero/logs/ \
        --randomize_seed False \
        --data_collection False \
        --available_gpus "0" \
        --seed 195 \
        --use_variance_scale False \
        --deterministic True \
        --run_id_note "reproduction--seed195--rtxpro6000" \
        --ar_future_prediction False \
        --ar_value_prediction False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 2>&1 | tee "$LOG_DIR/${suite}_output.log"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "=== Finished $suite in ${ELAPSED}s: $(date) ===" | tee -a "$LOG_DIR/eval_summary.log"
done

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true
echo "" | tee -a "$LOG_DIR/eval_summary.log"
echo "=== All LIBERO Evaluations Completed: $(date) ===" | tee -a "$LOG_DIR/eval_summary.log"
