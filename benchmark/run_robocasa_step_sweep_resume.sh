#!/bin/bash
# RoboCasa step sweep - RESUME
# 1-step: PnPCounterToCab(29/50 done) + PnPCabToCounter(13/50 done) already complete
#         Resume from PnPCounterToSink onwards
# 3-step: All 24 tasks (all failed previously due to NVML error)
set -e
cd /workspace

TASKS_1STEP_RESUME=(
    "PnPCounterToSink" "PnPSinkToCounter"
    "PnPCounterToMicrowave" "PnPMicrowaveToCounter" "PnPCounterToStove" "PnPStoveToCounter"
    "OpenSingleDoor" "CloseSingleDoor" "OpenDoubleDoor" "CloseDoubleDoor"
    "OpenDrawer" "CloseDrawer"
    "TurnOnStove" "TurnOffStove"
    "TurnOnSinkFaucet" "TurnOffSinkFaucet" "TurnSinkSpout"
    "CoffeeSetupMug" "CoffeeServeMug" "CoffeePressButton"
    "TurnOnMicrowave" "TurnOffMicrowave"
)

TASKS_ALL=(
    "PnPCounterToCab" "PnPCabToCounter" "PnPCounterToSink" "PnPSinkToCounter"
    "PnPCounterToMicrowave" "PnPMicrowaveToCounter" "PnPCounterToStove" "PnPStoveToCounter"
    "OpenSingleDoor" "CloseSingleDoor" "OpenDoubleDoor" "CloseDoubleDoor"
    "OpenDrawer" "CloseDrawer"
    "TurnOnStove" "TurnOffStove"
    "TurnOnSinkFaucet" "TurnOffSinkFaucet" "TurnSinkSpout"
    "CoffeeSetupMug" "CoffeeServeMug" "CoffeePressButton"
    "TurnOnMicrowave" "TurnOffMicrowave"
)

run_task() {
    local task=$1
    local n_steps=$2
    local log_dir="benchmark/results/robocasa_step_sweep/${n_steps}step"
    mkdir -p "$log_dir"

    echo "=== Starting $task (${n_steps}-step): $(date) ===" | tee -a "$log_dir/eval_summary.log"
    local start=$(date +%s)

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
        --run_id_note "${n_steps}step--seed195--exp4" \
        --local_log_dir cosmos_policy/experiments/robot/robocasa/logs/ \
        --seed 195 \
        --randomize_seed False \
        --deterministic True \
        --use_variance_scale False \
        --use_jpeg_compression True \
        --flip_images True \
        --num_denoising_steps_action "$n_steps" \
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --data_collection False 2>&1 | tee "$log_dir/${task}_output.log"

    local elapsed=$(( $(date +%s) - start ))
    # extract success count
    local succ
    succ=$(grep -oP '(?<=Successes: )[0-9]+(?=/)' "$log_dir/${task}_output.log" 2>/dev/null | tail -1 || true)
    [ -z "$succ" ] && succ=$(grep 'success rate' "$log_dir/${task}_output.log" 2>/dev/null | tail -1 || echo "?")
    echo "=== Finished $task (${n_steps}-step) in ${elapsed}s: ${succ}/50 ===" | tee -a "$log_dir/eval_summary.log"
}

# 1-step: resume from PnPCounterToSink
LOG_1STEP="benchmark/results/robocasa_step_sweep/1step/eval_summary.log"
mkdir -p benchmark/results/robocasa_step_sweep/1step
echo "" | tee -a "$LOG_1STEP"
echo "=== 1-step RESUME started: $(date) ===" | tee -a "$LOG_1STEP"

for task in "${TASKS_1STEP_RESUME[@]}"; do
    run_task "$task" 1
done

# Summarize 1-step (all 24 including previously completed)
echo "" | tee -a "$LOG_1STEP"
echo "=== 1-step sweep complete: $(date) ===" | tee -a "$LOG_1STEP"
uv run --extra cu128 --group robocasa --python 3.10 python -c "
import os, re, glob
log_dir = 'benchmark/results/robocasa_step_sweep/1step'
tasks = [
    'PnPCounterToCab','PnPCabToCounter','PnPCounterToSink','PnPSinkToCounter',
    'PnPCounterToMicrowave','PnPMicrowaveToCounter','PnPCounterToStove','PnPStoveToCounter',
    'OpenSingleDoor','CloseSingleDoor','OpenDoubleDoor','CloseDoubleDoor',
    'OpenDrawer','CloseDrawer','TurnOnStove','TurnOffStove',
    'TurnOnSinkFaucet','TurnOffSinkFaucet','TurnSinkSpout',
    'CoffeeSetupMug','CoffeeServeMug','CoffeePressButton',
    'TurnOnMicrowave','TurnOffMicrowave'
]
total, succ = 0, 0
print('=== 1-step per-task results ===')
for t in tasks:
    log = os.path.join(log_dir, f'{t}_output.log')
    if not os.path.exists(log):
        print(f'{t}: MISSING')
        continue
    content = open(log).read()
    # find final success rate
    m = re.findall(r'Success rate: ([0-9.]+)', content) or re.findall(r'([0-9]+)/50 success', content)
    sr = m[-1] if m else '?'
    # find raw counts
    counts = re.findall(r'([0-9]+)/50', content)
    raw = counts[-1] if counts else '?'
    print(f'{t}: {raw}/50')
    if counts:
        succ += int(counts[-1])
    total += 50
print(f'Total 1-step: {succ}/{total} = {succ*100/total:.1f}%')
" 2>/dev/null | tee -a "$LOG_1STEP"

# 3-step: all 24 tasks
LOG_3STEP="benchmark/results/robocasa_step_sweep/3step/eval_summary.log"
mkdir -p benchmark/results/robocasa_step_sweep/3step
echo "" | tee -a "$LOG_3STEP"
echo "=== 3-step started: $(date) ===" | tee -a "$LOG_3STEP"

for task in "${TASKS_ALL[@]}"; do
    run_task "$task" 3
done

echo "" | tee -a "$LOG_3STEP"
echo "=== 3-step sweep complete: $(date) ===" | tee -a "$LOG_3STEP"
uv run --extra cu128 --group robocasa --python 3.10 python -c "
import os, re
log_dir = 'benchmark/results/robocasa_step_sweep/3step'
tasks = [
    'PnPCounterToCab','PnPCabToCounter','PnPCounterToSink','PnPSinkToCounter',
    'PnPCounterToMicrowave','PnPMicrowaveToCounter','PnPCounterToStove','PnPStoveToCounter',
    'OpenSingleDoor','CloseSingleDoor','OpenDoubleDoor','CloseDoubleDoor',
    'OpenDrawer','CloseDrawer','TurnOnStove','TurnOffStove',
    'TurnOnSinkFaucet','TurnOffSinkFaucet','TurnSinkSpout',
    'CoffeeSetupMug','CoffeeServeMug','CoffeePressButton',
    'TurnOnMicrowave','TurnOffMicrowave'
]
total, succ = 0, 0
print('=== 3-step per-task results ===')
for t in tasks:
    log = os.path.join(log_dir, f'{t}_output.log')
    if not os.path.exists(log):
        print(f'{t}: MISSING')
        continue
    content = open(log).read()
    counts = re.findall(r'([0-9]+)/50', content)
    raw = counts[-1] if counts else '?'
    print(f'{t}: {raw}/50')
    if counts:
        succ += int(counts[-1])
    total += 50
print(f'Total 3-step: {succ}/{total} = {succ*100/total:.1f}%')
" 2>/dev/null | tee -a "$LOG_3STEP"

echo "=== ALL DONE ==="
