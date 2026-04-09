#!/bin/bash
# Experiment 0-A: Collect candidate verification data for rank correlation analysis
# GPU constraint: ONLY uses GPUs 3 and 4 (parallel: tasks 0-4 on GPU3, tasks 5-9 on GPU4)
# WARNING: Do NOT touch other GPUs — other users' processes may be running on them.
#
# Usage:
#   cd /home/jongholee/research/cosmos-policy
#   bash cosmos_policy/experiments/robot/analysis/run_exp0a.sh
#
# Expected runtime: ~2-3 hours (parallel across 2 GPUs)
# Output: ../../adaptive-ttc-wam/experiments/0A_rank_correlation/

set -euo pipefail

OUTPUT_DIR="../../adaptive-ttc-wam/experiments/0A_rank_correlation"
mkdir -p "$OUTPUT_DIR"

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
    --task_suite_name libero_spatial
    --num_trials_per_task 10
    --seed 195
    --deterministic True
    --use_jpeg_compression True
    --flip_images True
    --num_denoising_steps_action 1
    --num_denoising_steps_future_state 1
    --num_denoising_steps_value 1
    --num_candidates 8
    --num_wm_rollouts_full 3
    --num_value_preds_full 5
    --output_dir "$OUTPUT_DIR"
)

echo "============================================"
echo "Exp 0-A: Shallow-Full Rank Correlation"
echo "GPU 3: Tasks 0-4  |  GPU 4: Tasks 5-9"
echo "Date: $(date)"
echo "============================================"

# Launch GPU 3: Tasks 0-4
echo "[GPU 3] Starting tasks 0-4..."
CUDA_VISIBLE_DEVICES=0 uv run --extra cu128 --group libero -m cosmos_policy.experiments.robot.analysis.collect_candidate_data \
    "${COMMON_ARGS[@]}" \
    --task_start 0 --task_end 5 \
    > "$OUTPUT_DIR/gpu3_tasks0-4.log" 2>&1 &
PID_GPU3=$!

# Launch GPU 4: Tasks 5-9
echo "[GPU 4] Starting tasks 5-9..."
CUDA_VISIBLE_DEVICES=1 uv run --extra cu128 --group libero -m cosmos_policy.experiments.robot.analysis.collect_candidate_data \
    "${COMMON_ARGS[@]}" \
    --task_start 5 --task_end 10 \
    > "$OUTPUT_DIR/gpu4_tasks5-9.log" 2>&1 &
PID_GPU4=$!

echo "PIDs: GPU3=$PID_GPU3, GPU4=$PID_GPU4"
echo "Logs: $OUTPUT_DIR/gpu3_tasks0-4.log, $OUTPUT_DIR/gpu4_tasks5-9.log"
echo ""
echo "Waiting for both jobs to complete..."
echo "(Monitor with: tail -f $OUTPUT_DIR/gpu3_tasks0-4.log)"

# Wait for both to finish
FAIL=0
wait $PID_GPU3 || { echo "[GPU 3] FAILED (exit code $?)"; FAIL=1; }
wait $PID_GPU4 || { echo "[GPU 4] FAILED (exit code $?)"; FAIL=1; }

if [ $FAIL -ne 0 ]; then
    echo "ERROR: One or both jobs failed. Check logs above."
    exit 1
fi

echo ""
echo "============================================"
echo "Data collection complete! Merging results..."
echo "============================================"

# Merge the two JSON files into one
python3 -c "
import json, glob, os, sys

output_dir = '$OUTPUT_DIR'
merged_path = os.path.join(output_dir, 'candidate_data_libero_spatial_merged.json')

# Remove any stale merged file so it cannot be picked up by the glob below
if os.path.exists(merged_path):
    os.remove(merged_path)
    print(f'Removed stale merged file: {merged_path}')

# Only match per-shard files; the merged file is already excluded by the removal above,
# but the explicit exclusion guard here makes the intent clear.
all_candidates = sorted(glob.glob(os.path.join(output_dir, 'candidate_data_libero_spatial*.json')))
files = [f for f in all_candidates if not f.endswith('_merged.json')]

if len(files) < 2:
    print(f'ERROR: Expected at least 2 shard files, found {len(files)}: {files}')
    sys.exit(1)

print(f'Merging {len(files)} files: {[os.path.basename(f) for f in files]}')

# Load all shards and validate before merging
shards = []
for f_path in files:
    with open(f_path) as f:
        shards.append((f_path, json.load(f)))

# Validate: all shards must share the same config
ref_config = shards[0][1].get('config')
for f_path, data in shards[1:]:
    shard_config = data.get('config')
    if shard_config != ref_config:
        print(f'ERROR: Config mismatch between {os.path.basename(files[0])} and {os.path.basename(f_path)}')
        print(f'  ref  : {ref_config}')
        print(f'  found: {shard_config}')
        sys.exit(1)

# Validate: task_ids must be non-overlapping across shards
seen_task_ids = set()
for f_path, data in shards:
    task_ids = {t['task_id'] for t in data['tasks']}
    overlap = seen_task_ids & task_ids
    if overlap:
        print(f'ERROR: Duplicate task_ids {sorted(overlap)} found in {os.path.basename(f_path)}')
        sys.exit(1)
    seen_task_ids.update(task_ids)

# Merge
merged = None
for f_path, data in shards:
    if merged is None:
        merged = data
    else:
        merged['tasks'].extend(data['tasks'])
        merged['summary']['total_episodes'] += data['summary']['total_episodes']
        merged['summary']['total_successes'] += data['summary']['total_successes']
        merged['summary']['total_decision_points'] += data['summary']['total_decision_points']

# Recompute success rate
total_ep = merged['summary']['total_episodes']
total_succ = merged['summary']['total_successes']
merged['summary']['success_rate'] = total_succ / total_ep if total_ep > 0 else 0

# Sort tasks by task_id
merged['tasks'].sort(key=lambda t: t['task_id'])

with open(merged_path, 'w') as f:
    json.dump(merged, f, indent=2)

print(f'Merged: {len(merged[\"tasks\"])} tasks, {total_ep} episodes, {merged[\"summary\"][\"total_decision_points\"]} decision points')
print(f'Success rate: {merged[\"summary\"][\"success_rate\"]:.1%}')
print(f'Saved to: {merged_path}')
"

MERGED_FILE="$OUTPUT_DIR/candidate_data_libero_spatial_merged.json"

echo ""
echo "Running Exp 0-A analysis (rank correlation)..."
uv run --extra cu128 --group libero -m cosmos_policy.experiments.robot.analysis.rank_correlation_analysis \
    --input_path "$MERGED_FILE" \
    --output_dir "$OUTPUT_DIR"

echo ""
echo "Running Exp 0-B analysis (uncertainty-reversal)..."
uv run --extra cu128 --group libero -m cosmos_policy.experiments.robot.analysis.uncertainty_reversal_analysis \
    --input_path "$MERGED_FILE" \
    --output_dir "../../adaptive-ttc-wam/experiments/0B_uncertainty_reversal"

echo ""
echo "============================================"
echo "All analyses complete!"
echo "Results:"
echo "  0-A: $OUTPUT_DIR/rank_correlation_results.json"
echo "  0-B: ../../adaptive-ttc-wam/experiments/0B_uncertainty_reversal/uncertainty_reversal_results.json"
echo "============================================"
