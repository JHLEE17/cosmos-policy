#!/bin/bash
# Experiment 0-Zero Phase A: Value Stability with 5-step Denoising
# Tests whether increasing denoising steps (1→5) fixes the value noise problem.
#
# Key difference from run_exp0a.sh:
#   - num_denoising_steps_future_state: 1 → 5
#   - num_denoising_steps_value: 1 → 5  (linspace: each WM rollout gets [1,2,3,4,5]-step values)
#   - num_denoising_steps_action stays at 1 (cheap proposals, matching research design)
#
# GPU: ONLY GPU 3
# Expected runtime: ~4-6 hours (single GPU, higher denoising steps = slower)
#
# Usage:
#   cd /home/jongholee/research/cosmos-policy
#   bash cosmos_policy/experiments/robot/analysis/run_exp0zero_phase_a.sh

set -euo pipefail

# ==========================================
# Environment setup (user venv + nvidia libs)
# ==========================================
VENV_USER="/home/jongholee/research/cosmos-policy/.venv-user"
NVIDIA_BASE="/home/jongholee/research/cosmos-policy/.venv/lib/python3.10/site-packages/nvidia"
SHIM_DIR="$VENV_USER/lib/nvidia-libs"
ALL_NVIDIA_LIBS=$(find "$NVIDIA_BASE" -name "lib" -type d | paste -sd:)

export PATH="$SHIM_DIR:$PATH"
export LD_LIBRARY_PATH="$SHIM_DIR:$ALL_NVIDIA_LIBS"
export PYTHONPATH="$VENV_USER/sitecustomize"
export CUDA_VISIBLE_DEVICES=3

PYTHON="$VENV_USER/bin/python3"

OUTPUT_DIR="../../adaptive-ttc-wam/experiments/0Zero_value_validity/phase_a_5step"
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
    --num_denoising_steps_future_state 5
    --num_denoising_steps_value 5
    --num_candidates 8
    --num_wm_rollouts_full 3
    --num_value_preds_full 5
    --output_dir "$OUTPUT_DIR"
)

echo "============================================"
echo "Exp 0-Zero Phase A: 5-step Denoising"
echo "  future_state denoising: 5 steps"
echo "  value denoising: 5 steps (linspace [1,2,3,4,5])"
echo "  action denoising: 1 step (cheap proposals)"
echo "GPU 3 only (sequential: tasks 0-4, then 5-9)"
echo "Date: $(date)"
echo "============================================"

# Run batch 1: tasks 0-4
echo "[GPU 3] Starting tasks 0-4..."
$PYTHON -c "import usercustomize" 2>/dev/null  # pre-apply patch
$PYTHON -m cosmos_policy.experiments.robot.analysis.collect_candidate_data \
    "${COMMON_ARGS[@]}" \
    --task_start 0 --task_end 5 \
    > "$OUTPUT_DIR/gpu3_tasks0-4.log" 2>&1
echo "[BATCH 1] Complete!"

# Run batch 2: tasks 5-9
echo "[GPU 3] Starting tasks 5-9..."
$PYTHON -m cosmos_policy.experiments.robot.analysis.collect_candidate_data \
    "${COMMON_ARGS[@]}" \
    --task_start 5 --task_end 10 \
    > "$OUTPUT_DIR/gpu3_tasks5-9.log" 2>&1
echo "[BATCH 2] Complete!"

echo ""
echo "============================================"
echo "Data collection complete! Merging results..."
echo "============================================"

# Merge the two JSON files
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

print(f'Merging {len(files)} files: {[os.path.basename(f) for f in files]}')

shards = []
for f_path in files:
    with open(f_path) as f:
        shards.append((f_path, json.load(f)))

seen_task_ids = set()
for f_path, data in shards:
    task_ids = {t['task_id'] for t in data['tasks']}
    overlap = seen_task_ids & task_ids
    if overlap:
        print(f'ERROR: Duplicate task_ids {sorted(overlap)} found in {os.path.basename(f_path)}')
        sys.exit(1)
    seen_task_ids.update(task_ids)

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

print(f'Merged: {len(merged[\"tasks\"])} tasks, {total_ep} episodes, {merged[\"summary\"][\"total_decision_points\"]} decision points')
print(f'Success rate: {merged[\"summary\"][\"success_rate\"]:.1%}')
print(f'Saved to: {merged_path}')
"

MERGED_FILE="$OUTPUT_DIR/candidate_data_libero_spatial_merged.json"
ANALYSIS_OUTPUT="../../adaptive-ttc-wam/experiments/0Zero_value_validity"

echo ""
echo "Running 0-Zero-A: Value Stability Analysis..."
$PYTHON -m cosmos_policy.experiments.robot.analysis.value_stability_analysis \
    --input_path "$MERGED_FILE" \
    --output_dir "${ANALYSIS_OUTPUT}/phase_a_5step_results"

echo ""
echo "Running 0-Zero-B: Value-Accuracy Analysis..."
$PYTHON -m cosmos_policy.experiments.robot.analysis.value_accuracy_analysis \
    --input_path "$MERGED_FILE" \
    --output_dir "${ANALYSIS_OUTPUT}/phase_a_5step_results"

echo ""
echo "Running Exp 0-A: Rank Correlation (for comparison with 1-step)..."
$PYTHON -m cosmos_policy.experiments.robot.analysis.rank_correlation_analysis \
    --input_path "$MERGED_FILE" \
    --output_dir "${ANALYSIS_OUTPUT}/phase_a_5step_results"

echo ""
echo "Running Exp 0-B: Uncertainty Reversal (for comparison with 1-step)..."
$PYTHON -m cosmos_policy.experiments.robot.analysis.uncertainty_reversal_analysis \
    --input_path "$MERGED_FILE" \
    --output_dir "${ANALYSIS_OUTPUT}/phase_a_5step_results"

echo ""
echo "============================================"
echo "Phase A Complete!"
echo "Results in: ${ANALYSIS_OUTPUT}/phase_a_5step_results"
echo "============================================"
