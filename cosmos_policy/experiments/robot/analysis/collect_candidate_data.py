# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Experiment 0-A: Collect per-candidate shallow and full verification data for rank correlation analysis.

For each decision state in LIBERO episodes:
1. Generate N action candidates (1-step denoising, different seeds) via get_action()
2. For each candidate, run AR verification pipeline:
   - Shallow: 1 WM rollout -> 1 value prediction
   - Full: 3 WM rollouts -> 5 value predictions per rollout (15 total)
3. Save ALL individual value estimates to JSON for downstream analysis (0-A, 0-B)

GPU constraint: Use ONLY GPUs 3,4 (set CUDA_VISIBLE_DEVICES=3 before running).

Usage:
    CUDA_VISIBLE_DEVICES=3 uv run -m cosmos_policy.experiments.robot.analysis.collect_candidate_data \
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
        --num_denoising_steps_future_state 1 \
        --num_denoising_steps_value 1 \
        --num_candidates 8 \
        --num_wm_rollouts_full 3 \
        --num_value_preds_full 5 \
        --output_dir ../../adaptive-ttc-wam/experiments/0A_rank_correlation
"""

import json
import logging
import os
import secrets
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import draccus
import numpy as np
import torch
import tqdm

from cosmos_policy.experiments.robot.cosmos_utils import (
    extract_value_from_latent_sequence,
    get_action,
    get_future_state_prediction,
    get_model,
    get_planning_model,
    init_t5_text_embeddings_cache,
    load_dataset_stats,
)
from cosmos_policy.experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
)
from cosmos_policy.experiments.robot.robot_utils import (
    DATE_TIME,
    get_image_resize_size,
    log_message,
    setup_logging,
)
from cosmos_policy.utils.utils import set_seed_everywhere


# Task suite definitions (same as run_libero_eval.py)
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"


TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,
    TaskSuite.LIBERO_OBJECT: 280,
    TaskSuite.LIBERO_GOAL: 300,
    TaskSuite.LIBERO_10: 520,
    TaskSuite.LIBERO_90: 400,
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    # fmt: off
    suite: str = "libero"

    # Cosmos Policy parameters
    model_family: str = "cosmos"
    config: str = ""
    ckpt_path: str = ""
    planning_model_config_name: str = ""
    planning_model_ckpt_path: str = ""
    config_file: str = "cosmos_policy/config/config.py"

    use_third_person_image: bool = True
    num_third_person_images: int = 1
    use_wrist_image: bool = True
    num_wrist_images: int = 1
    use_proprio: bool = True
    flip_images: bool = True
    use_variance_scale: bool = False
    use_jpeg_compression: bool = True
    num_denoising_steps_action: int = 1      # 1-step for cheap proposals
    num_denoising_steps_future_state: int = 1
    num_denoising_steps_value: int = 1
    unnormalize_actions: bool = True
    normalize_proprio: bool = True
    dataset_stats_path: str = ""
    t5_text_embeddings_path: str = ""
    trained_with_image_aug: bool = True
    chunk_size: int = 16
    num_open_loop_steps: int = 16

    deterministic: bool = True
    deterministic_reset: bool = False
    deterministic_reset_seed: int = None

    # LIBERO-specific
    task_suite_name: str = TaskSuite.LIBERO_SPATIAL
    num_trials_per_task: int = 10
    initial_states_path: str = "DEFAULT"
    env_img_res: int = 256

    seed: int = 195
    randomize_seed: bool = False

    # =============================================
    # Analysis-specific parameters
    # =============================================
    num_candidates: int = 8                  # N in Best-of-N
    num_wm_rollouts_full: int = 3            # WM rollouts for full verification
    num_value_preds_full: int = 5            # Value predictions per WM rollout for full verification
    output_dir: str = "../../adaptive-ttc-wam/experiments/0A_rank_correlation"
    task_start: int = -1                     # First task index to run (-1 = all tasks)
    task_end: int = -1                       # Last task index (exclusive) (-1 = all tasks)

    # Flags not used but needed for cosmos_utils compatibility
    ar_future_prediction: bool = True        # Must be True for separate WM/value calls
    ar_value_prediction: bool = True         # Must be True for separate value calls
    ar_qvalue_prediction: bool = False
    use_ensemble_future_state_predictions: bool = False
    num_future_state_predictions_in_ensemble: int = 1
    future_state_ensemble_aggregation_scheme: str = "average"
    use_ensemble_value_predictions: bool = False
    num_value_predictions_in_ensemble: int = 1
    value_ensemble_aggregation_scheme: str = "average"
    mask_current_state_action_for_value_prediction: bool = False
    mask_future_state_for_qvalue_prediction: bool = False
    search_depth: int = 1
    search_depth_value_aggregation_scheme: str = "use_last_value"
    use_parallel_inference: bool = False
    available_gpus: str = "3,4"
    parallel_timeout: int = 15
    num_queries_best_of_n: int = 1
    data_collection: bool = False
    # fmt: on


def prepare_observation(obs, resize_size, flip_images: bool = False):
    """Prepare observation for policy input."""
    img = get_libero_image(obs, flip_images)
    wrist_img = get_libero_wrist_image(obs, flip_images)
    observation = {
        "primary_image": img,
        "wrist_image": wrist_img,
        "proprio": np.concatenate((obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"])),
    }
    return observation


def collect_individual_values(
    cfg,
    model,
    data_batch,
    future_state_samples_list,
    seed,
    num_value_predictions,
    num_denoising_steps_value=1,
):
    """
    Generate individual value predictions without aggregation.

    For each future state sample, generates `num_value_predictions` value predictions
    with different seeds. Returns ALL individual values.

    Returns:
        list[float]: All individual value estimates (len = len(future_state_samples_list) * num_value_predictions)
    """
    all_values = []

    with torch.inference_mode():
        data_batch["num_conditional_frames"] = model.config.state_t - 1
        data_batch["mask_current_state_action_for_value_prediction"] = (
            cfg.mask_current_state_action_for_value_prediction
        )

        for fs_idx, fs_sample in enumerate(future_state_samples_list):
            # Generate value predictions with different seeds and denoising steps
            value_seeds = [seed + fs_idx * 100 + i for i in range(num_value_predictions)]
            value_denoising_steps_list = (
                np.linspace(1, num_denoising_steps_value, num_value_predictions).astype(int).tolist()
            )

            for v_idx in range(num_value_predictions):
                generated_latent_with_value = model.generate_samples_from_batch(
                    data_batch,
                    n_sample=1,
                    num_steps=value_denoising_steps_list[v_idx],
                    seed=value_seeds[v_idx],
                    is_negative_prompt=False,
                    use_variance_scale=cfg.use_variance_scale,
                    skip_vae_encoding=True,
                    previous_generated_latent=fs_sample,
                )
                value_indices = torch.full(
                    (1,), -1, dtype=torch.int64, device=generated_latent_with_value.device
                )
                value_prediction = extract_value_from_latent_sequence(generated_latent_with_value, value_indices)
                value_prediction = (value_prediction + 1) / 2
                value_prediction = torch.clamp(value_prediction, min=0, max=1)
                all_values.append(value_prediction[0].item())

    return all_values


def collect_candidate_verification_data(
    cfg,
    model,
    planning_model,
    dataset_stats,
    observation,
    task_description,
    candidate_seed,
    num_wm_rollouts_full,
    num_value_preds_full,
):
    """
    For a single action candidate, collect both shallow and full verification data.

    Returns dict with:
    - shallow_value: single value from 1 WM rollout + 1 value prediction
    - full_values: list of all individual values from full verification
    - full_value_mean: mean of full_values
    - full_value_std: std of full_values
    """
    active_model = planning_model if planning_model is not None else model

    # Step 1: Generate action candidate (1-step denoising for cheap proposal)
    action_return_dict = get_action(
        cfg,
        model,
        dataset_stats,
        observation,
        task_description,
        seed=candidate_seed,
        randomize_seed=False,
        num_denoising_steps_action=cfg.num_denoising_steps_action,
        generate_future_state_and_value_in_parallel=False,  # AR mode
    )

    # Step 2: Generate future state predictions (WM rollouts)
    # Full verification: num_wm_rollouts_full WM rollouts
    full_fs_return = get_future_state_prediction(
        cfg,
        model=active_model,
        data_batch=action_return_dict["data_batch"],
        generated_latent_with_action=action_return_dict["generated_latent"],
        orig_clean_latent_frames=action_return_dict["orig_clean_latent_frames"],
        future_proprio_latent_idx=action_return_dict["latent_indices"]["future_proprio_latent_idx"],
        future_wrist_image_latent_idx=action_return_dict["latent_indices"]["future_wrist_image_latent_idx"],
        future_wrist_image2_latent_idx=action_return_dict["latent_indices"]["future_wrist_image2_latent_idx"],
        future_image_latent_idx=action_return_dict["latent_indices"]["future_image_latent_idx"],
        future_image2_latent_idx=action_return_dict["latent_indices"]["future_image2_latent_idx"],
        seed=candidate_seed,
        randomize_seed=False,
        num_denoising_steps_future_state=cfg.num_denoising_steps_future_state,
        use_ensemble_future_state_predictions=True,
        num_future_state_predictions_in_ensemble=num_wm_rollouts_full,
        future_state_ensemble_aggregation_scheme="average",
    )

    full_fs_samples = full_fs_return["future_state_samples_list"]

    # Step 3: Collect ALL individual value predictions from full verification
    full_values = collect_individual_values(
        cfg,
        model=active_model,
        data_batch=action_return_dict["data_batch"],
        future_state_samples_list=full_fs_samples,
        seed=candidate_seed,
        num_value_predictions=num_value_preds_full,
        num_denoising_steps_value=cfg.num_denoising_steps_value,
    )

    # Shallow value = first WM rollout, first value prediction = full_values[0]
    shallow_value = full_values[0]

    # Full aggregated value = mean of all values
    full_value_mean = float(np.mean(full_values))
    full_value_std = float(np.std(full_values))

    return {
        "seed": candidate_seed,
        "shallow_value": shallow_value,
        "full_values": full_values,
        "full_value_mean": full_value_mean,
        "full_value_std": full_value_std,
        "actions": [a.tolist() if isinstance(a, np.ndarray) else a for a in action_return_dict["actions"]],
    }


def run_analysis_episode(
    cfg: AnalysisConfig,
    env,
    task_description: str,
    model,
    planning_model,
    dataset_stats,
    resize_size,
    initial_state=None,
    log_file=None,
):
    """Run a single episode collecting per-candidate verification data at each decision point."""
    from libero.libero import benchmark

    # Reset environment
    if cfg.deterministic_reset:
        reset_seed = cfg.deterministic_reset_seed if cfg.deterministic_reset_seed is not None else cfg.seed
        set_seed_everywhere(reset_seed)
    env.reset()

    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue = deque(maxlen=cfg.num_open_loop_steps)
    t = 0
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Data collection for analysis
    decision_points = []  # List of per-decision-point candidate data
    success = False

    try:
        NUM_STEPS_WAIT = 10
        while t < max_steps + NUM_STEPS_WAIT:
            if os.environ.get("DETERMINISTIC", "").lower() == "true":
                set_seed_everywhere(0)

            # Wait for objects to stabilize
            if t < NUM_STEPS_WAIT:
                obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                t += 1
                continue

            # Prepare observation
            observation = prepare_observation(obs, resize_size, cfg.flip_images)

            # If action queue is empty, collect candidate data
            if len(action_queue) == 0:
                decision_point_data = {
                    "timestep": t,
                    "episode_phase": "early" if t < max_steps * 0.33 else ("mid" if t < max_steps * 0.66 else "late"),
                    "candidates": [],
                }

                log_message(f"t={t}: Collecting {cfg.num_candidates} candidates...", log_file)
                start_time = time.time()

                # Generate and verify N candidates
                for cand_idx in range(cfg.num_candidates):
                    cand_seed = cfg.seed + cand_idx
                    cand_data = collect_candidate_verification_data(
                        cfg=cfg,
                        model=model,
                        planning_model=planning_model,
                        dataset_stats=dataset_stats,
                        observation=observation,
                        task_description=task_description,
                        candidate_seed=cand_seed,
                        num_wm_rollouts_full=cfg.num_wm_rollouts_full,
                        num_value_preds_full=cfg.num_value_preds_full,
                    )
                    # Don't save raw actions to JSON (too large)
                    actions_for_execution = cand_data.pop("actions")
                    decision_point_data["candidates"].append(cand_data)

                    # Track best candidate for episode execution
                    if cand_idx == 0 or cand_data["full_value_mean"] > best_value:
                        best_value = cand_data["full_value_mean"]
                        best_actions = actions_for_execution

                collection_time = time.time() - start_time
                decision_point_data["collection_time_sec"] = collection_time

                log_message(
                    f"t={t}: Collected {cfg.num_candidates} candidates in {collection_time:.1f}s. "
                    f"Best value: {best_value:.4f}",
                    log_file,
                )

                decision_points.append(decision_point_data)

                # Execute best action to continue the episode
                action_queue.extend(best_actions)

            # Execute action
            action = action_queue.popleft()
            obs, reward, done, info = env.step(action.tolist() if isinstance(action, np.ndarray) else action)
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        error_msg = f"Episode error: {e}"
        traceback_str = traceback.format_exc()
        log_message(f"{error_msg}\nFull traceback:\n{traceback_str}", log_file)

    return success, decision_points


@draccus.wrap()
def collect_data(cfg: AnalysisConfig):
    """Main function: collect per-candidate verification data across LIBERO episodes."""
    from libero.libero import benchmark

    logger.info(f"GPU constraint: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    logger.info(f"Using device: cuda:0 (mapped to physical GPU by CUDA_VISIBLE_DEVICES)")

    # Deterministic mode
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"

    # Validate
    assert cfg.ckpt_path, "ckpt_path must not be empty!"

    # Set seed
    set_seed_everywhere(cfg.seed)

    # Initialize model
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, cosmos_config = get_model(cfg)

    # Set unnorm key (used as index into dataset_stats)
    cfg.unnorm_key = cfg.task_suite_name

    # Planning model (if separate)
    planning_model = None
    if cfg.planning_model_ckpt_path:
        planning_model, _ = get_planning_model(cfg)

    resize_size = get_image_resize_size(cfg.model_family)

    # Setup output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Setup logging
    log_filepath = os.path.join(cfg.output_dir, f"collection_log_{DATE_TIME}.txt")
    log_file = open(log_filepath, "w")
    log_message(f"Analysis config: {cfg}", log_file)

    # Initialize LIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # Apply task range filter
    task_start = cfg.task_start if cfg.task_start >= 0 else 0
    task_end = cfg.task_end if cfg.task_end >= 0 else num_tasks

    log_message(f"Task suite: {cfg.task_suite_name}, {num_tasks} total tasks, running [{task_start}, {task_end})", log_file)
    log_message(f"Candidates per state: {cfg.num_candidates}", log_file)
    log_message(f"Full verification: {cfg.num_wm_rollouts_full} WM x {cfg.num_value_preds_full} V", log_file)

    # Collect data across all tasks and episodes
    all_results = {
        "config": {
            "task_suite": cfg.task_suite_name,
            "num_candidates": cfg.num_candidates,
            "num_wm_rollouts_full": cfg.num_wm_rollouts_full,
            "num_value_preds_full": cfg.num_value_preds_full,
            "num_denoising_steps_action": cfg.num_denoising_steps_action,
            "num_denoising_steps_future_state": cfg.num_denoising_steps_future_state,
            "num_denoising_steps_value": cfg.num_denoising_steps_value,
            "seed": cfg.seed,
            "num_trials_per_task": cfg.num_trials_per_task,
        },
        "tasks": [],
    }

    total_decision_points = 0
    total_episodes = 0
    total_successes = 0

    for task_id in tqdm.tqdm(range(task_start, task_end), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)

        task_data = {
            "task_id": task_id,
            "task_description": task_description,
            "episodes": [],
        }

        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task), desc=f"Task {task_id} episodes", leave=False):
            initial_state = initial_states[episode_idx]
            log_message(f"\nTask {task_id} ({task_description}), Episode {episode_idx + 1}", log_file)

            success, decision_points = run_analysis_episode(
                cfg=cfg,
                env=env,
                task_description=task_description,
                model=model,
                planning_model=planning_model,
                dataset_stats=dataset_stats,
                resize_size=resize_size,
                initial_state=initial_state,
                log_file=log_file,
            )

            episode_data = {
                "episode_idx": episode_idx,
                "success": success,
                "num_decision_points": len(decision_points),
                "decision_points": decision_points,
            }
            task_data["episodes"].append(episode_data)

            total_decision_points += len(decision_points)
            total_episodes += 1
            total_successes += int(success)

            log_message(
                f"Episode {episode_idx + 1}: {'SUCCESS' if success else 'FAIL'}, "
                f"{len(decision_points)} decision points collected",
                log_file,
            )

            # Save incrementally after each episode
            if (episode_idx + 1) % 5 == 0 or episode_idx == cfg.num_trials_per_task - 1:
                # Save task-level checkpoint
                checkpoint_path = os.path.join(cfg.output_dir, f"task{task_id}_checkpoint.json")
                with open(checkpoint_path, "w") as f:
                    json.dump(task_data, f, indent=2)

        all_results["tasks"].append(task_data)

        # Log task summary
        task_episodes = len(task_data["episodes"])
        task_successes = sum(1 for ep in task_data["episodes"] if ep["success"])
        task_sr = task_successes / task_episodes if task_episodes > 0 else 0
        log_message(
            f"Task {task_id} ({task_description}): {task_successes}/{task_episodes} = {task_sr:.1%}",
            log_file,
        )

        # Cleanup env
        env.close()

    # Save final results
    all_results["summary"] = {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_decision_points": total_decision_points,
        "success_rate": total_successes / total_episodes if total_episodes > 0 else 0,
    }

    task_range_str = f"_tasks{task_start}-{task_end}" if (task_start > 0 or task_end < num_tasks) else ""
    output_path = os.path.join(cfg.output_dir, f"candidate_data_{cfg.task_suite_name}{task_range_str}_{DATE_TIME}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    log_message(f"\nData collection complete!", log_file)
    log_message(f"Total episodes: {total_episodes}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Total decision points: {total_decision_points}", log_file)
    log_message(f"Success rate: {total_successes / total_episodes:.1%}", log_file)
    log_message(f"Output saved to: {output_path}", log_file)

    log_file.close()
    return output_path


if __name__ == "__main__":
    collect_data()
