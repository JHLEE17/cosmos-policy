#!/bin/bash
# RoboCasa denoising step sweep: 1-step and 3-step
# 5-step results already available from reproduction run (65.6%)
# Run inside cosmos-exp-jongholee container: docker exec cosmos-exp-jongholee bash benchmark/run_robocasa_step_sweep.sh
set -e
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

STEPS=(1 3)

for N_STEPS in "${STEPS[@]}"; do
    LOG_DIR="benchmark/results/robocasa_step_sweep/${N_STEPS}step"
    mkdir -p "$LOG_DIR"

    echo "=== RoboCasa ${N_STEPS}-step Sweep Started: $(date) ===" | tee "$LOG_DIR/eval_summary.log"
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" | tee -a "$LOG_DIR/eval_summary.log"
    echo "Denoising steps (action): ${N_STEPS}" | tee -a "$LOG_DIR/eval_summary.log"
    TOTAL_SUCC=0
    TOTAL_TRIALS=0

    for task in "${TASKS[@]}"; do
        echo "" | tee -a "$LOG_DIR/eval_summary.log"
        echo "=== Starting $task (${N_STEPS}-step): $(date) ===" | tee -a "$LOG_DIR/eval_summary.log"
        START_TIME=$(date +%s)

        uv run --extra cu128 --group robocasa --python 3.10 \
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
            --run_id_note "${N_STEPS}step--seed195--exp4" \
            --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
            --seed 195 \
            --randomize_seed False \
            --deterministic True \
            --use_variance_scale False \
            --use_jpeg_compression True \
            --flip_images True \
            --num_denoising_steps_action "$N_STEPS" \
            --num_denoising_steps_future_state 1 \
            --num_denoising_steps_value 1 \
            --data_collection False 2>&1 | tee "$LOG_DIR/${task}_output.log"

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        # Extract success count from log
        SUCC=$(grep -oP '(?<=Successes: )[0-9]+(?=/)' "$LOG_DIR/${task}_output.log" | tail -1 || echo "0")
        echo "=== Finished $task in ${ELAPSED}s: ${SUCC}/50 successes ===" | tee -a "$LOG_DIR/eval_summary.log"
        TOTAL_SUCC=$((TOTAL_SUCC + SUCC))
        TOTAL_TRIALS=$((TOTAL_TRIALS + 50))
    done

    SR=$(awk "BEGIN {printf \"%.1f\", $TOTAL_SUCC * 100 / $TOTAL_TRIALS}")
    echo "" | tee -a "$LOG_DIR/eval_summary.log"
    echo "=== ${N_STEPS}-step COMPLETE: ${TOTAL_SUCC}/${TOTAL_TRIALS} = ${SR}% ===" | tee -a "$LOG_DIR/eval_summary.log"
    echo "=== Finished: $(date) ===" | tee -a "$LOG_DIR/eval_summary.log"
done

echo ""
echo "=== All RoboCasa Step Sweep Done ==="
