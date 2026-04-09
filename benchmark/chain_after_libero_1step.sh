#!/bin/bash
# Chain: wait for LIBERO 1-step to finish → extract results → start RoboCasa sweep
set -e
cd /workspace

LIBERO_LOG="cosmos_policy/experiments/robot/libero/logs/ENV_EVAL-libero_spatial-cosmos-2026_03_24-01_03_02--1step--seed195--exp3-resume.txt"
SUMMARY="benchmark/results/experiment_summary.md"

echo "[chain] Waiting for LIBERO 1-step eval to complete (500 episodes)..."

# Poll until done
while true; do
    TOTAL=$(grep '# episodes completed so far:' "$LIBERO_LOG" 2>/dev/null | tail -1 | grep -oP '[0-9]+' | head -1)
    echo "[chain] $(date '+%H:%M:%S'): ${TOTAL:-0}/500 episodes"
    if [ "${TOTAL:-0}" -ge 500 ]; then
        break
    fi
    sleep 120
done

echo "[chain] LIBERO 1-step complete! Extracting results..."

# Extract per-task success rates
TASK_RATES=$(grep 'Current task success rate:' "$LIBERO_LOG" | grep -oP '[0-9.]+')
TOTAL_RATE=$(grep 'Current total success rate:' "$LIBERO_LOG" | tail -1 | grep -oP '[0-9.]+')
TOTAL_RATE_PCT=$(echo "scale=1; $TOTAL_RATE * 100" | bc)

echo "[chain] Per-task rates: $TASK_RATES"
echo "[chain] Total: ${TOTAL_RATE_PCT}%"

# Write results JSON
mkdir -p benchmark/results/libero_steps_pareto
python3 -c "
import json, re

log = open('$LIBERO_LOG').read()
task_rates = [float(x) for x in re.findall(r'Current task success rate: ([0-9.]+)', log)]
total_rate = float(re.findall(r'Current total success rate: ([0-9.]+)', log)[-1])

result = {
    'steps': 1,
    'task_suite': 'libero_spatial',
    'seed': 195,
    'per_task_success_rates': task_rates,
    'total_success_rate': total_rate,
    'total_success_pct': round(total_rate * 100, 1),
    'num_tasks': len(task_rates),
    'episodes_per_task': 50,
}
out = 'benchmark/results/libero_steps_pareto/1step_results.json'
with open(out, 'w') as f:
    json.dump(result, f, indent=2)
print(f'Saved to {out}')
print(f'Per-task: {[round(r*100) for r in task_rates]}')
print(f'Total: {result[\"total_success_pct\"]}%')
"

echo "[chain] Starting RoboCasa step sweep (1-step + 3-step)..."
bash benchmark/run_robocasa_step_sweep.sh

echo "[chain] All done!"
