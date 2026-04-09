#!/bin/bash
cd /workspace

TASKS=(
    "PnPCounterToCab" "PnPCabToCounter" "PnPCounterToSink" "PnPSinkToCounter"
    "PnPCounterToMicrowave" "PnPMicrowaveToCounter" "PnPCounterToStove" "PnPStoveToCounter"
    "OpenSingleDoor" "CloseSingleDoor" "OpenDoubleDoor" "CloseDoubleDoor"
    "OpenDrawer" "CloseDrawer"
    "TurnOnStove" "TurnOffStove"
    "TurnOnSinkFaucet" "TurnOffSinkFaucet" "TurnSinkSpout"
    "CoffeeSetupMug" "CoffeeServeMug" "CoffeePressButton"
    "TurnOnMicrowave" "TurnOffMicrowave"
)

LOG_DIR="benchmark/robocasa_results"
mkdir -p "$LOG_DIR"

# Start GPU monitoring
bash benchmark/monitor_gpu.sh "$LOG_DIR/gpu_monitor.csv" &
GPU_MONITOR_PID=$!

echo "=== RoboCasa Evaluation Started: $(date) ===" | tee "$LOG_DIR/eval_summary.log"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee -a "$LOG_DIR/eval_summary.log"
echo "Total tasks: ${#TASKS[@]}" | tee -a "$LOG_DIR/eval_summary.log"

for task in "${TASKS[@]}"; do
    echo "" | tee -a "$LOG_DIR/eval_summary.log"
    echo "=== Starting $task: $(date) ===" | tee -a "$LOG_DIR/eval_summary.log"
    START_TIME=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=0 uv run --extra cu128 --group robocasa --python 3.10 \
      python -m cosmos_policy.experiments.robot.robocasa.run_robocasa_eval \
        --config cosmos_predict2_2b_480p_robocasa_50_demos_per_task__inference \
        --ckpt_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B \
        --config_file cosmos_policy/config/config.py \
        --use_wrist_image True \
        --num_wrist_images 1 \
        --use_proprio True \
        --normalize_proprio True \
        --unnormalize_actions True \
        --dataset_stats_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_dataset_statistics.json \
        --t5_text_embeddings_path nvidia/Cosmos-Policy-RoboCasa-Predict2-2B/robocasa_t5_embeddings.pkl \
        --trained_with_image_aug True \
        --chunk_size 32 \
        --num_open_loop_steps 16 \
        --task_name "$task" \
        --num_trials_per_task 50 \
        --run_id_note "reproduction--seed195--rtxpro6000" \
        --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
        --seed 195 \
        --randomize_seed False \
        --deterministic True \
        --use_variance_scale False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action 5 \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --data_collection False 2>&1 | tee "$LOG_DIR/${task}_output.log"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "=== Finished $task in ${ELAPSED}s: $(date) ===" | tee -a "$LOG_DIR/eval_summary.log"
done

# Stop GPU monitoring
kill $GPU_MONITOR_PID 2>/dev/null || true
echo "" | tee -a "$LOG_DIR/eval_summary.log"
echo "=== All RoboCasa Evaluations Completed: $(date) ===" | tee -a "$LOG_DIR/eval_summary.log"
