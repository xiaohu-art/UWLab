# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to collect demonstrations from a trained RL policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import contextlib
import gymnasium as gym
import os
import time
import torch
from tqdm import tqdm

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect demonstrations from trained RL policy.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--dataset_file", type=str, default="./datasets/dataset.zarr", help="Output dataset path.")
parser.add_argument("--num_demos", type=int, default=10, help="Number of demonstrations to record.")
parser.add_argument(
    "--deterministic",
    action="store_true",
    default=False,
    help="Use the mean of the policy distribution instead of sampling.",
)
parser.add_argument(
    "--stats_interval_s",
    type=float,
    default=0.0,
    help="Print rollout throughput and termination diagnostics every N seconds. Disabled by default.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.managers.recorder_manager import DatasetExportMode

# Import dataset handlers
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

from uwlab.utils.datasets import ZarrDatasetFileHandler

import uwlab_tasks  # noqa: F401
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from uwlab_tasks.utils.hydra import hydra_task_compose

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

WARMUP_STEPS = 2


def process_agent_cfg(env_cfg, agent_cfg):
    if hasattr(agent_cfg.algorithm, "behavior_cloning_cfg"):
        if agent_cfg.algorithm.behavior_cloning_cfg is None:
            del agent_cfg.algorithm.behavior_cloning_cfg
        else:
            bc_cfg = agent_cfg.algorithm.behavior_cloning_cfg
            if bc_cfg.experts_observation_group_cfg is not None:
                import importlib

                # resolve path to the module location
                mod_name, attr_name = bc_cfg.experts_observation_group_cfg.split(":")
                mod = importlib.import_module(mod_name)
                cfg_cls = mod
                for attr in attr_name.split("."):
                    cfg_cls = getattr(cfg_cls, attr)
                cfg = cfg_cls()
                setattr(env_cfg.observations, "expert_obs", cfg)

    if hasattr(agent_cfg.algorithm, "offline_algorithm_cfg"):
        if agent_cfg.algorithm.offline_algorithm_cfg is None:
            del agent_cfg.algorithm.offline_algorithm_cfg
        else:
            if agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg is None:
                del agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
            else:
                bc_cfg = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
                if bc_cfg.experts_observation_group_cfg is not None:
                    import importlib

                    # resolve path to the module location
                    mod_name, attr_name = bc_cfg.experts_observation_group_cfg.split(":")
                    mod = importlib.import_module(mod_name)
                    cfg_cls = mod
                    for attr in attr_name.split("."):
                        cfg_cls = getattr(cfg_cls, attr)
                    cfg = cfg_cls()
                    setattr(env_cfg.observations, "expert_obs", cfg)
    return agent_cfg


@hydra_task_compose(args_cli.task, "rsl_rl_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect demonstrations from the environment using RSL-RL policy."""
    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.basename(args_cli.dataset_file)

    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # add recordermanager to save data
    use_zarr_format = args_cli.dataset_file.endswith(".zarr")
    if use_zarr_format:
        dataset_handler = ZarrDatasetFileHandler
    else:
        dataset_handler = HDF5DatasetFileHandler

    # Setup recorder for raw actions
    env_cfg.recorders = ActionStateRecorderManagerCfg()

    env_cfg.recorders.dataset_export_dir_path = output_dir
    env_cfg.recorders.dataset_filename = output_file_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    env_cfg.recorders.dataset_file_handler_class_type = dataset_handler

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = None

    # add expert obs into env_cfg
    agent_cfg = process_agent_cfg(env_cfg, agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load expert
    bc = agent_cfg.algorithm.offline_algorithm_cfg.behavior_cloning_cfg
    assert len(bc.experts_path) == 1, "Only one expert is supported for now."
    expert_obs_fn = bc.experts_observation_func
    loader = bc.experts_loader
    if not callable(loader):
        loader = eval(loader)
    expert_policy = loader(bc.experts_path[0]).to(env_cfg.sim.device)
    expert_policy.eval()

    print(f"[Policy] {'Deterministic (mean)' if args_cli.deterministic else 'Stochastic (sampled)'} actions")

    term_names = getattr(env.unwrapped.termination_manager, "_term_names", [])
    term_counts = {name: 0 for name in term_names}
    total_finished_episodes = 0
    total_steps = 0
    stats_start_time = time.monotonic()
    last_stats_time = stats_start_time
    last_stats_steps = 0
    last_stats_finished_episodes = 0
    last_stats_recorded_demo_count = 0

    # simulate environment -- run everything in inference mode
    current_recorded_demo_count = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        # Initialize tqdm progress bar if num_demos > 0
        pbar = tqdm(total=args_cli.num_demos, desc="Recording Demonstrations", unit="demo")

        while True:
            # agent stepping
            expert_policy_obs = expert_obs_fn(env)
            mean, std = expert_policy.compute_distribution(expert_policy_obs)
            actions = mean if args_cli.deterministic else torch.normal(mean, std)

            warmup_mask = env.unwrapped.episode_length_buf < WARMUP_STEPS
            if torch.any(warmup_mask):
                actions[warmup_mask, :-1] = 0.0
                actions[warmup_mask, -1] = -1.0  # close gripper

            # Inject expert distribution into obs_buf so recorder saves them alongside observations
            env.unwrapped.obs_buf["data_collection"]["expert_action_mean"] = mean.clone()
            env.unwrapped.obs_buf["data_collection"]["expert_action_std"] = std.clone()

            # env stepping
            step_result = env.step(actions)
            total_steps += 1

            dones = None
            if isinstance(step_result, tuple):
                if len(step_result) == 4:
                    _, _, dones, _ = step_result
                elif len(step_result) == 5:
                    _, _, terminated, truncated, _ = step_result
                    dones = terminated | truncated

            if isinstance(dones, torch.Tensor) and dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                total_finished_episodes += len(reset_ids)
                term_dones = env.unwrapped.termination_manager._term_dones[reset_ids]
                for term_row in term_dones:
                    active = term_row.nonzero(as_tuple=False).flatten().cpu().tolist()
                    for term_idx in active:
                        if term_idx < len(term_names):
                            term_counts[term_names[term_idx]] += 1

            # print out the current demo count if it has changed
            new_count = env.unwrapped.recorder_manager.exported_successful_episode_count
            if new_count > current_recorded_demo_count:
                increment = new_count - current_recorded_demo_count
                current_recorded_demo_count = new_count
                pbar.update(increment)

            if args_cli.stats_interval_s > 0:
                now = time.monotonic()
                if now - last_stats_time >= args_cli.stats_interval_s:
                    window_s = max(now - last_stats_time, 1e-6)
                    elapsed_s = max(now - stats_start_time, 1e-6)
                    window_steps = total_steps - last_stats_steps
                    window_finished = total_finished_episodes - last_stats_finished_episodes
                    window_recorded = current_recorded_demo_count - last_stats_recorded_demo_count
                    success_count = term_counts.get("success", 0)
                    success_rate = success_count / total_finished_episodes if total_finished_episodes > 0 else 0.0
                    term_summary = ", ".join(f"{name}={count}" for name, count in term_counts.items() if count)
                    if not term_summary:
                        term_summary = "none"
                    tqdm.write(
                        "[stats] "
                        f"recorded={current_recorded_demo_count}/{args_cli.num_demos} "
                        f"finished={total_finished_episodes} "
                        f"success_rate={success_rate:.1%} "
                        f"steps/s={total_steps / elapsed_s:.2f} "
                        f"window_steps/s={window_steps / window_s:.2f} "
                        f"window_finished={window_finished} "
                        f"window_recorded={window_recorded} "
                        f"terms=({term_summary})"
                    )
                    last_stats_time = now
                    last_stats_steps = total_steps
                    last_stats_finished_episodes = total_finished_episodes
                    last_stats_recorded_demo_count = current_recorded_demo_count

            if args_cli.num_demos > 0 and new_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations recorded. Exiting the app.")
                break

            # check that simulation is stopped or not
            if env.unwrapped.sim.is_stopped():
                break

        pbar.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function - the decorator handles parameter passing
    main()  # type: ignore
    # close sim app
    simulation_app.close()
