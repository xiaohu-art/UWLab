# Copyright (c) 2024-2026, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Roll out an OpenPI policy server in the OmniReset RGB peg environment.

This is a lightweight sanity/evaluation script for the pi0.5 policy fine-tuned on
the LeRobot dataset exported from OmniReset. It does not modify the data
collection pipeline and does not record training data.

Expected OpenPI observation contract:
    observation/front_image: uint8 HWC RGB, 224x224x3
    observation/side_image:  uint8 HWC RGB, 224x224x3
    observation/wrist_image: uint8 HWC RGB, 224x224x3
    observation/state:       float32, shape (19,)

The 19-D state is concatenated in the exact order used by zarr_to_lerobot.py:
    last_gripper_action, last_arm_action, arm_joint_pos, end_effector_pose
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time
from typing import Any

import numpy as np
import torch
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Roll out an OpenPI server policy in OmniReset peg insertion.")
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task id.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel envs. Keep 1 for first sanity checks.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of finished episodes to evaluate.")
parser.add_argument("--max_steps", type=int, default=250, help="Safety cap on env steps, across all envs.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--host", type=str, default="127.0.0.1", help="OpenPI websocket server host.")
parser.add_argument("--port", type=int, default=8000, help="OpenPI websocket server port.")
parser.add_argument("--prompt", type=str, default="peg_insertion")
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=2,
    help="Initial zero-arm/closed-gripper steps after reset. Matches the successful-demo collector warmup.",
)
parser.add_argument(
    "--openpi_client_src",
    type=Path,
    default=None,
    help="Path to openpi/packages/openpi-client/src if openpi_client is not installed in the UWLab env.",
)
parser.add_argument("--save_video", action="store_true", help="Save concatenated front|side|wrist rollout video.")
parser.add_argument("--video_path", type=Path, default=Path("openpi_omnireset_rollout.mp4"))
parser.add_argument("--video_fps", type=int, default=10)
parser.add_argument("--print_actions", action="store_true", help="Print the first returned action at every policy query.")

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import imageio
import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectRLEnvCfg, ManagerBasedRLEnvCfg
from tqdm import tqdm

import uwlab_tasks  # noqa: F401
from uwlab_tasks.utils.hydra import hydra_task_compose


STATE_KEYS = ("last_gripper_action", "last_arm_action", "arm_joint_pos", "end_effector_pose")
IMAGE_KEYS = ("front_rgb", "side_rgb", "wrist_rgb")


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _candidate_openpi_client_paths() -> list[Path]:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = []
    if args_cli.openpi_client_src is not None:
        candidates.append(args_cli.openpi_client_src)
    env_path = os_environ_path("OPENPI_CLIENT_SRC")
    if env_path is not None:
        candidates.append(env_path)
    # Local layout used on the A100 machine: /.../ReAlign/UWLab and /.../ReAlign/external/openpi.
    candidates.append(repo_root.parent / "external" / "openpi" / "packages" / "openpi-client" / "src")
    # 5090 layout used in this project: /.../UWLab and /.../ReAlign/external/openpi.
    candidates.append(repo_root.parent / "ReAlign" / "external" / "openpi" / "packages" / "openpi-client" / "src")
    return candidates


def os_environ_path(name: str) -> Path | None:
    import os

    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def _import_openpi_client():
    for candidate in _candidate_openpi_client_paths():
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            break
    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise ImportError(
            "Could not import openpi_client. Either install the openpi client in this UWLab environment, or pass "
            "--openpi_client_src /path/to/openpi/packages/openpi-client/src."
        ) from exc
    return websocket_client_policy


def _to_numpy(x: Any, env_idx: int) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x[env_idx].detach().cpu().numpy()
    else:
        x = np.asarray(x)[env_idx]
    return np.asarray(x)


def _rgb_from_data_collection(obs_group: dict[str, Any], key: str, env_idx: int) -> np.ndarray:
    img = _to_numpy(obs_group[key], env_idx)
    if img.ndim != 3:
        raise RuntimeError(f"{key} should be a 3D image, got shape {img.shape}.")
    if img.shape[0] in (1, 3, 4) and img.shape[0] < img.shape[1]:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0.0, 1.0) if np.issubdtype(img.dtype, np.floating) else img
        img = (img * 255.0).clip(0, 255).astype(np.uint8) if np.issubdtype(img.dtype, np.floating) else img.astype(np.uint8)
    return np.ascontiguousarray(img)


def _state_from_data_collection(obs_group: dict[str, Any], env_idx: int) -> np.ndarray:
    parts = []
    for key in STATE_KEYS:
        value = _to_numpy(obs_group[key], env_idx).astype(np.float32, copy=False).reshape(-1)
        parts.append(value)
    state = np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    if state.shape != (19,):
        raise RuntimeError(f"Expected 19-D OpenPI state, got shape {state.shape}.")
    return state


def _make_openpi_obs(obs_group: dict[str, Any], env_idx: int) -> dict[str, Any]:
    return {
        "observation/front_image": _rgb_from_data_collection(obs_group, "front_rgb", env_idx),
        "observation/side_image": _rgb_from_data_collection(obs_group, "side_rgb", env_idx),
        "observation/wrist_image": _rgb_from_data_collection(obs_group, "wrist_rgb", env_idx),
        "observation/state": _state_from_data_collection(obs_group, env_idx),
        "prompt": args_cli.prompt,
    }


def _action_from_server(policy, obs_group: dict[str, Any], num_envs: int, device: torch.device) -> torch.Tensor:
    actions = torch.zeros((num_envs, 7), dtype=torch.float32, device=device)
    for env_idx in range(num_envs):
        out = policy.infer(_make_openpi_obs(obs_group, env_idx))
        chunk = np.asarray(out["actions"], dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[1] != 7:
            raise RuntimeError(f"Expected OpenPI action chunk shape (H, 7), got {chunk.shape}.")
        if args_cli.print_actions:
            print(f"[action env={env_idx}] {chunk[0]}")
        actions[env_idx] = torch.as_tensor(chunk[0], dtype=torch.float32, device=device)
    return actions


def _warmup_action(num_envs: int, device: torch.device) -> torch.Tensor:
    action = torch.zeros((num_envs, 7), dtype=torch.float32, device=device)
    action[:, -1] = -1.0
    return action


def _count_successes(env, reset_ids: torch.Tensor, term_names: list[str]) -> int:
    count = 0
    term_dones = env.unwrapped.termination_manager._term_dones[reset_ids]
    for term_row in term_dones:
        active = term_row.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        if any(term_idx < len(term_names) and term_names[term_idx] == "success" for term_idx in active):
            count += 1
    return count


def _capture_concat_frame(obs_group: dict[str, Any], env_idx: int = 0) -> np.ndarray:
    return np.concatenate([_rgb_from_data_collection(obs_group, key, env_idx) for key in IMAGE_KEYS], axis=1)


def _obs_group(env, obs_dict: dict[str, Any]) -> dict[str, Any]:
    if isinstance(obs_dict, dict) and "data_collection" in obs_dict:
        return obs_dict["data_collection"]
    if "data_collection" in env.unwrapped.obs_buf:
        return env.unwrapped.obs_buf["data_collection"]
    raise RuntimeError("No data_collection observation group found. Use an RGB-DataCollection OmniReset task.")


@hydra_task_compose(args_cli.task, "env_cfg_entry_point", hydra_args=remaining_args)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg):  # noqa: ARG001
    _set_seeds(args_cli.seed)

    device = torch.device(args_cli.device if args_cli.device else "cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.seed = args_cli.seed
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.concatenate_terms = False
    if hasattr(env_cfg.observations, "data_collection"):
        env_cfg.observations.data_collection.concatenate_terms = False

    websocket_client_policy = _import_openpi_client()
    policy = websocket_client_policy.WebsocketClientPolicy(host=args_cli.host, port=args_cli.port)
    print(f"[OpenPI] server metadata: {policy.get_server_metadata()}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    obs_dict, _ = env.reset()

    term_names = list(getattr(env.unwrapped.termination_manager, "_term_names", []))
    print(f"[Env] terminations: {term_names}")

    episodes = 0
    successes = 0
    total_steps = 0
    episode_steps = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    frames = []
    pbar = tqdm(total=args_cli.num_episodes, desc="OpenPI rollout")

    try:
        while simulation_app.is_running() and episodes < args_cli.num_episodes and total_steps < args_cli.max_steps:
            obs_group = _obs_group(env, obs_dict)
            if args_cli.save_video:
                frames.append(_capture_concat_frame(obs_group, env_idx=0))

            if torch.any(episode_steps < args_cli.warmup_steps):
                actions = _action_from_server(policy, obs_group, args_cli.num_envs, device)
                warmup_mask = episode_steps < args_cli.warmup_steps
                actions[warmup_mask] = _warmup_action(args_cli.num_envs, device)[warmup_mask]
            else:
                actions = _action_from_server(policy, obs_group, args_cli.num_envs, device)

            step_result = env.step(actions)
            if len(step_result) == 4:
                obs_dict, rewards, dones, infos = step_result
            else:
                obs_dict, rewards, terminated, truncated, infos = step_result
                dones = terminated | truncated

            total_steps += 1
            episode_steps += 1

            if isinstance(dones, torch.Tensor) and dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                successes += _count_successes(env, reset_ids, term_names)
                episodes += int(reset_ids.numel())
                episode_steps[reset_ids] = 0
                pbar.update(int(reset_ids.numel()))
                pbar.set_postfix(success=f"{successes}/{episodes}", rate=f"{successes / max(episodes, 1):.1%}")

        print("\nFinal Statistics:")
        print(f"episodes={episodes}, successes={successes}, success_rate={successes / max(episodes, 1):.2%}")
        print(f"env_steps={total_steps}")
        if args_cli.save_video and frames:
            args_cli.video_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(args_cli.video_path, frames, fps=args_cli.video_fps, codec="libx264")
            print(f"Saved video: {args_cli.video_path}")
    finally:
        pbar.close()
        env.close()


if __name__ == "__main__":
    main()  # type: ignore
    simulation_app.close()
