"""Merge all task shard JSONs + checkpoints into one merged file.

Usage:
    python -m cosmos_policy.experiments.robot.analysis.merge_all_tasks \
        --output_dir /path/to/output --suite libero_spatial
"""
import argparse
import json
import os
import glob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--suite", default="libero_spatial")
    args = parser.parse_args()

    output_dir = args.output_dir
    suite = args.suite

    merged_path = os.path.join(output_dir, f"candidate_data_{suite}_merged.json")
    if os.path.exists(merged_path):
        os.remove(merged_path)

    # Load final JSON shard files
    final_files = sorted(glob.glob(os.path.join(output_dir, f"candidate_data_{suite}*.json")))
    final_files = [f for f in final_files if not f.endswith("_merged.json")]
    print(f"Shard files: {[os.path.basename(f) for f in final_files]}")

    all_tasks = {}
    config = None
    for f_path in final_files:
        with open(f_path) as f:
            data = json.load(f)
        if config is None:
            config = data.get("config")
        for t in data.get("tasks", []):
            tid = t["task_id"]
            if tid not in all_tasks:
                all_tasks[tid] = t
                print(f"  From {os.path.basename(f_path)}: task {tid}")

    # Load checkpoints for missing tasks
    for tid in range(10):
        if tid not in all_tasks:
            cp_path = os.path.join(output_dir, f"task{tid}_checkpoint.json")
            if os.path.exists(cp_path):
                with open(cp_path) as f:
                    cp = json.load(f)
                all_tasks[tid] = cp
                n_eps = len(cp.get("episodes", []))
                print(f"  From checkpoint: task {tid} ({n_eps} episodes)")

    if not all_tasks:
        print("ERROR: No task data found")
        return

    print(f"\nTotal tasks: {len(all_tasks)}")
    print(f"Task IDs: {sorted(all_tasks.keys())}")

    # Compute summary
    total_episodes = 0
    total_successes = 0
    total_decision_points = 0
    for tid in sorted(all_tasks.keys()):
        t = all_tasks[tid]
        eps = t.get("episodes", [])
        total_episodes += len(eps)
        total_successes += sum(1 for e in eps if e.get("success", False))
        total_decision_points += sum(len(e.get("decision_points", [])) for e in eps)

    merged = {
        "config": config,
        "tasks": [all_tasks[tid] for tid in sorted(all_tasks.keys())],
        "summary": {
            "total_episodes": total_episodes,
            "total_successes": total_successes,
            "total_decision_points": total_decision_points,
            "success_rate": total_successes / total_episodes if total_episodes > 0 else 0,
        },
    }

    with open(merged_path, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"\nMerged: {len(merged['tasks'])} tasks, {total_episodes} episodes, {total_decision_points} decision points")
    print(f"Success rate: {merged['summary']['success_rate']:.1%}")
    print(f"Saved to: {merged_path}")


if __name__ == "__main__":
    main()
