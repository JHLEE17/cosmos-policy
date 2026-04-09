#!/bin/bash
# Wait for batch 1 (PID from arg), then run batch 2 + analyses
# Usage: bash run_phase_a_batch2_and_analyze.sh <batch1_pid>

set -euo pipefail

BATCH1_PID=${1:?Usage: $0 <batch1_pid>}

NVIDIA_BASE="/home/jongholee/research/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia"
SHIM_DIR="/home/jongholee/research/cosmos-policy/.venv-user/lib/nvidia-libs"
ALL_NVIDIA_LIBS=$(find "$NVIDIA_BASE" -name "lib" -type d | paste -sd:)
OUTPUT_DIR="/home/jongholee/research/adaptive-ttc-wam/experiments/0Zero_value_validity/phase_a_5step"
ANALYSIS_OUTPUT="/home/jongholee/research/adaptive-ttc-wam/experiments/0Zero_value_validity/phase_a_5step_results"
PYTHON="/home/jongholee/research/cosmos-policy/.venv-user/bin/python3"

export PATH="$SHIM_DIR:$PATH"
export LD_LIBRARY_PATH="$SHIM_DIR:$ALL_NVIDIA_LIBS"
export CUDA_VISIBLE_DEVICES=3

mkdir -p "$OUTPUT_DIR" "$ANALYSIS_OUTPUT"

echo "[$(date)] Waiting for batch 1 (PID=$BATCH1_PID) to finish..."
while kill -0 "$BATCH1_PID" 2>/dev/null; do sleep 30; done
echo "[$(date)] Batch 1 finished!"

echo "[$(date)] Starting batch 2 (tasks 5-9)..."
$PYTHON -m cosmos_policy.experiments.robot.analysis.collect_candidate_data \
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
    --task_suite_name libero_spatial \
    --num_trials_per_task 10 \
    --seed 195 \
    --deterministic True \
    --use_jpeg_compression True \
    --flip_images True \
    --num_denoising_steps_action 1 \
    --num_denoising_steps_future_state 5 \
    --num_denoising_steps_value 5 \
    --num_candidates 8 \
    --num_wm_rollouts_full 3 \
    --num_value_preds_full 5 \
    --output_dir "$OUTPUT_DIR" \
    --task_start 5 --task_end 10 \
    > "$OUTPUT_DIR/gpu3_tasks5-9.log" 2>&1

echo "[$(date)] Batch 2 complete! Merging..."

# Merge shards
$PYTHON -c "
import json, glob, os, sys

output_dir = '$OUTPUT_DIR'
merged_path = os.path.join(output_dir, 'candidate_data_libero_spatial_merged.json')

if os.path.exists(merged_path):
    os.remove(merged_path)

all_candidates = sorted(glob.glob(os.path.join(output_dir, 'candidate_data_libero_spatial*.json')))
files = [f for f in all_candidates if not f.endswith('_merged.json')]

if len(files) < 2:
    print(f'ERROR: Expected at least 2 shard files, found {len(files)}: {files}')
    sys.exit(1)

print(f'Merging {len(files)} files')
shards = []
for f_path in files:
    with open(f_path) as f:
        shards.append((f_path, json.load(f)))

merged = None
for f_path, data in shards:
    if merged is None:
        merged = data
    else:
        merged['tasks'].extend(data['tasks'])
        merged['summary']['total_episodes'] += data['summary']['total_episodes']
        merged['summary']['total_successes'] += data['summary']['total_successes']
        merged['summary']['total_decision_points'] += data['summary']['total_decision_points']

total_ep = merged['summary']['total_episodes']
total_succ = merged['summary']['total_successes']
merged['summary']['success_rate'] = total_succ / total_ep if total_ep > 0 else 0
merged['tasks'].sort(key=lambda t: t['task_id'])

with open(merged_path, 'w') as f:
    json.dump(merged, f, indent=2)
print(f'Merged: {len(merged[\"tasks\"])} tasks, {total_ep} episodes')
"

MERGED_FILE="$OUTPUT_DIR/candidate_data_libero_spatial_merged.json"

echo "[$(date)] Running analyses..."
$PYTHON -m cosmos_policy.experiments.robot.analysis.value_stability_analysis \
    --input_path "$MERGED_FILE" --output_dir "$ANALYSIS_OUTPUT" 2>&1
$PYTHON -m cosmos_policy.experiments.robot.analysis.value_accuracy_analysis \
    --input_path "$MERGED_FILE" --output_dir "$ANALYSIS_OUTPUT" 2>&1
$PYTHON -m cosmos_policy.experiments.robot.analysis.rank_correlation_analysis \
    --input_path "$MERGED_FILE" --output_dir "$ANALYSIS_OUTPUT" 2>&1
$PYTHON -m cosmos_policy.experiments.robot.analysis.uncertainty_reversal_analysis \
    --input_path "$MERGED_FILE" --output_dir "$ANALYSIS_OUTPUT" 2>&1

echo "[$(date)] ALL DONE! Results in: $ANALYSIS_OUTPUT"
