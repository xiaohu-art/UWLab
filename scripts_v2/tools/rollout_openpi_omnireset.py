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
from collections import Counter
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
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
    "--replan_interval",
    type=int,
    default=10,
    help="How many actions to execute from each OpenPI action chunk before querying again.",
)
parser.add_argument(
    "--arm_action_scale",
    type=float,
    default=1.0,
    help="Post-policy multiplier for the first 6 arm action dimensions. Use <1 only for debugging unstable rollouts.",
)
parser.add_argument(
    "--action_clip",
    type=float,
    default=None,
    help="Optional symmetric clip applied to all 7 raw action dimensions after arm_action_scale.",
)
parser.add_argument(
    "--warmup_steps",
    type=int,
    default=2,
    help="Initial zero-arm/closed-gripper steps after reset. Matches the successful-demo collector warmup.",
)
parser.add_argument(
    "--reset_rerenders",
    type=int,
    default=8,
    help="Number of forced RTX rerenders on reset for evaluation rollouts.",
)
parser.add_argument(
    "--reset_preroll_steps",
    type=int,
    default=2,
    help=(
        "After each reset, step this many times with the fixed warmup action before recording video or querying "
        "OpenPI. This mirrors the demo conversion trim-head behavior and avoids using reset-transient camera frames."
    ),
)
parser.add_argument(
    "--disable_temporal_rendering",
    action="store_true",
    help="Disable temporal RTX render features such as DL denoiser / DLAA for debugging cross-episode ghosting.",
)
parser.add_argument(
    "--explicit_reset_between_episodes",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "After an episode terminates, discard the auto-reset observation returned by env.step() and call env.reset() "
        "explicitly before starting the next episode. Keep this on for camera-based rollout evaluation."
    ),
)
parser.add_argument(
    "--recreate_env_each_episode",
    action="store_true",
    help="Slow diagnostic mode: close and recreate the Isaac env/camera render products after every episode.",
)
parser.add_argument(
    "--openpi_client_src",
    type=Path,
    default=None,
    help="Path to openpi/packages/openpi-client/src if openpi_client is not installed in the UWLab env.",
)
parser.add_argument("--save_video", action="store_true", help="Save concatenated front|side|wrist rollout video.")
parser.add_argument(
    "--video_path",
    type=Path,
    default=Path("openpi_omnireset_rollout.mp4"),
    help=(
        "Episode video output path or directory. If this has a video suffix, per-episode files are written as "
        "<stem>_episode_XXXXXX<suffix>. If it has no suffix, files are written under that directory."
    ),
)
parser.add_argument("--video_fps", type=int, default=10)
parser.add_argument("--print_actions", action="store_true", help="Print the first returned action at every policy query.")

AppLauncher.add_app_launcher_args(parser)
args_cli, remaining_args = parser.parse_known_args()
if remaining_args:
    print(f"[Hydra overrides] {remaining_args}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import imageio
import isaaclab_tasks  # noqa: F401
import torch
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


def _action_chunk_from_server(policy, obs_group: dict[str, Any], env_idx: int) -> np.ndarray:
    out = policy.infer(_make_openpi_obs(obs_group, env_idx))
    chunk = np.asarray(out["actions"], dtype=np.float32)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise RuntimeError(f"Expected OpenPI action chunk shape (H, 7), got {chunk.shape}.")
    if args_cli.print_actions:
        print(f"[action env={env_idx}] {chunk[0]}")
    return chunk


def _actions_from_chunk_cache(
    policy,
    obs_group: dict[str, Any],
    episode_steps: torch.Tensor,
    action_chunks: list[np.ndarray | None],
    chunk_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    num_envs = len(action_chunks)
    actions = _warmup_action(num_envs, device)
    for env_idx in range(num_envs):
        if int(episode_steps[env_idx].item()) < args_cli.warmup_steps:
            continue
        chunk = action_chunks[env_idx]
        max_chunk_steps = args_cli.replan_interval
        if chunk is None or chunk_indices[env_idx] >= min(max_chunk_steps, len(chunk)):
            chunk = _action_chunk_from_server(policy, obs_group, env_idx)
            action_chunks[env_idx] = chunk
            chunk_indices[env_idx] = 0
        action = chunk[chunk_indices[env_idx]].copy()
        action[:6] *= args_cli.arm_action_scale
        if args_cli.action_clip is not None:
            action = np.clip(action, -args_cli.action_clip, args_cli.action_clip)
        actions[env_idx] = torch.as_tensor(action, dtype=torch.float32, device=device)
        chunk_indices[env_idx] += 1
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


def _termination_names_for_resets(env, reset_ids: torch.Tensor, term_names: list[str]) -> list[list[str]]:
    results = []
    term_dones = env.unwrapped.termination_manager._term_dones[reset_ids]
    for term_row in term_dones:
        active = term_row.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        names = [term_names[term_idx] for term_idx in active if term_idx < len(term_names)]
        results.append(names or ["unknown_done"])
    return results


def _capture_concat_frame(obs_group: dict[str, Any], env_idx: int = 0) -> np.ndarray:
    return np.concatenate([_rgb_from_data_collection(obs_group, key, env_idx) for key in IMAGE_KEYS], axis=1)


def _episode_video_path(base_path: Path, episode_idx: int) -> Path:
    if base_path.suffix:
        return base_path.with_name(f"{base_path.stem}_episode_{episode_idx:06d}{base_path.suffix}")
    return base_path / f"episode_{episode_idx:06d}.mp4"


def _save_episode_video(frames: list[np.ndarray], episode_idx: int) -> Path | None:
    if not frames:
        return None
    path = _episode_video_path(args_cli.video_path, episode_idx)
    if path.parent.exists() and not path.parent.is_dir():
        raise RuntimeError(
            f"Video output parent exists but is not a directory: {path.parent}. "
            "Choose a new --video_path directory, or remove/rename that file."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=args_cli.video_fps, codec="libx264")
    return path


def _step_env(env, actions: torch.Tensor):
    step_result = env.step(actions)
    if len(step_result) == 4:
        obs_dict, rewards, dones, infos = step_result
    else:
        obs_dict, rewards, terminated, truncated, infos = step_result
        dones = terminated | truncated
    return obs_dict, rewards, dones, infos


def _obs_group(env, obs_dict: dict[str, Any]) -> dict[str, Any]:
    if isinstance(obs_dict, dict) and "data_collection" in obs_dict:
        return obs_dict["data_collection"]
    if "data_collection" in env.unwrapped.obs_buf:
        return env.unwrapped.obs_buf["data_collection"]
    raise RuntimeError("No data_collection observation group found. Use an RGB-DataCollection OmniReset task.")


def _force_render_and_recompute_obs(env, num_renders: int):
    obs_dict = env.unwrapped.obs_buf
    for _ in range(max(num_renders, 0)):
        env.unwrapped.sim.render()
        obs_dict = env.unwrapped.observation_manager.compute(update_history=True)
        env.unwrapped.obs_buf = obs_dict
    return obs_dict


def _reset_eval_env(env, device: torch.device):
    obs_dict, _ = env.reset()
    obs_dict = _force_render_and_recompute_obs(env, args_cli.reset_rerenders)

    preroll_done = 0
    reset_attempts = 0
    while preroll_done < args_cli.reset_preroll_steps:
        obs_dict, _, dones, _ = _step_env(env, _warmup_action(args_cli.num_envs, device))
        if isinstance(dones, torch.Tensor) and dones.any():
            reset_attempts += 1
            if reset_attempts > 20:
                raise RuntimeError("Too many terminations during reset preroll; check reset/camera health.")
            obs_dict, _ = env.reset()
            obs_dict = _force_render_and_recompute_obs(env, args_cli.reset_rerenders)
            preroll_done = 0
            continue
        preroll_done += 1
    return obs_dict


def _configure_rendering_for_rollout(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg) -> None:
    if hasattr(env_cfg, "num_rerenders_on_reset"):
        env_cfg.num_rerenders_on_reset = args_cli.reset_rerenders
    if not args_cli.disable_temporal_rendering:
        return

    render_cfg = getattr(env_cfg.sim, "render", None)
    if render_cfg is None:
        return
    for name in ("enable_dl_denoiser", "enable_dlssg", "enable_reflections", "enable_ambient_occlusion"):
        if hasattr(render_cfg, name):
            setattr(render_cfg, name, False)
    if hasattr(render_cfg, "antialiasing_mode"):
        render_cfg.antialiasing_mode = "FXAA"


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
    _configure_rendering_for_rollout(env_cfg)

    websocket_client_policy = _import_openpi_client()
    policy = websocket_client_policy.WebsocketClientPolicy(host=args_cli.host, port=args_cli.port)
    print(f"[OpenPI] server metadata: {policy.get_server_metadata()}")

    def make_env_and_reset():
        new_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
        new_obs_dict = _reset_eval_env(new_env, device)
        return new_env, new_obs_dict

    env, obs_dict = make_env_and_reset()

    term_names = list(getattr(env.unwrapped.termination_manager, "_term_names", []))
    print(f"[Env] terminations: {term_names}")
    print(
        "[Eval reset] "
        f"explicit_reset_between_episodes={args_cli.explicit_reset_between_episodes}, "
        f"recreate_env_each_episode={args_cli.recreate_env_each_episode}, "
        f"reset_rerenders={args_cli.reset_rerenders}, "
        f"reset_preroll_steps={args_cli.reset_preroll_steps}, "
        f"disable_temporal_rendering={args_cli.disable_temporal_rendering}"
    )

    episodes = 0
    successes = 0
    total_steps = 0
    episode_steps = torch.full(
        (args_cli.num_envs,), args_cli.reset_preroll_steps, dtype=torch.long, device=device
    )
    action_chunks: list[np.ndarray | None] = [None for _ in range(args_cli.num_envs)]
    chunk_indices = [0 for _ in range(args_cli.num_envs)]
    termination_counts: Counter[str] = Counter()
    action_min = torch.full((7,), float("inf"), dtype=torch.float32, device=device)
    action_max = torch.full((7,), -float("inf"), dtype=torch.float32, device=device)
    action_abs_sum = torch.zeros((7,), dtype=torch.float32, device=device)
    action_count = 0
    episode_lengths: list[int] = []
    frames: list[np.ndarray] = []
    recorded_video_episodes = 0
    pbar = tqdm(total=args_cli.num_episodes, desc="OpenPI rollout")

    try:
        while simulation_app.is_running() and episodes < args_cli.num_episodes and total_steps < args_cli.max_steps:
            obs_group = _obs_group(env, obs_dict)
            if args_cli.save_video:
                frames.append(_capture_concat_frame(obs_group, env_idx=0))

            actions = _actions_from_chunk_cache(
                policy,
                obs_group,
                episode_steps,
                action_chunks,
                chunk_indices,
                device,
            )
            action_min = torch.minimum(action_min, actions.detach().amin(dim=0))
            action_max = torch.maximum(action_max, actions.detach().amax(dim=0))
            action_abs_sum += actions.detach().abs().sum(dim=0)
            action_count += actions.shape[0]

            obs_dict, rewards, dones, infos = _step_env(env, actions)

            total_steps += 1
            episode_steps += 1

            if isinstance(dones, torch.Tensor) and dones.any():
                reset_ids = (dones > 0).nonzero(as_tuple=False).reshape(-1)
                reset_term_names = _termination_names_for_resets(env, reset_ids, term_names)
                for names in reset_term_names:
                    termination_counts.update(names)
                successes += sum("success" in names for names in reset_term_names)
                episodes += int(reset_ids.numel())
                episode_lengths.extend([int(episode_steps[env_idx].item()) for env_idx in reset_ids])
                episode_steps[reset_ids] = 0
                if args_cli.save_video and any(int(env_idx) == 0 for env_idx in reset_ids.detach().cpu().tolist()):
                    video_path = _save_episode_video(frames, recorded_video_episodes)
                    if video_path is not None:
                        print(f"\nSaved video: {video_path}")
                        recorded_video_episodes += 1
                    frames = []
                for env_idx in reset_ids.detach().cpu().tolist():
                    action_chunks[env_idx] = None
                    chunk_indices[env_idx] = 0
                if args_cli.recreate_env_each_episode:
                    env.close()
                    env, obs_dict = make_env_and_reset()
                    term_names = list(getattr(env.unwrapped.termination_manager, "_term_names", []))
                    episode_steps[:] = args_cli.reset_preroll_steps
                    action_chunks = [None for _ in range(args_cli.num_envs)]
                    chunk_indices = [0 for _ in range(args_cli.num_envs)]
                elif args_cli.explicit_reset_between_episodes:
                    obs_dict = _reset_eval_env(env, device)
                    episode_steps[:] = args_cli.reset_preroll_steps
                    action_chunks = [None for _ in range(args_cli.num_envs)]
                    chunk_indices = [0 for _ in range(args_cli.num_envs)]
                pbar.update(int(reset_ids.numel()))
                recent_terms = "|".join(",".join(names) for names in reset_term_names)
                pbar.set_postfix(success=f"{successes}/{episodes}", rate=f"{successes / max(episodes, 1):.1%}", last=recent_terms)

        print("\nFinal Statistics:")
        print(f"episodes={episodes}, successes={successes}, success_rate={successes / max(episodes, 1):.2%}")
        print(f"env_steps={total_steps}")
        print(f"terminations={dict(termination_counts)}")
        if episode_lengths:
            print(
                "episode_lengths="
                f"min={min(episode_lengths)}, max={max(episode_lengths)}, "
                f"mean={sum(episode_lengths) / len(episode_lengths):.1f}"
            )
        if action_count > 0:
            action_mean_abs = action_abs_sum / action_count
            print(f"action_min={action_min.detach().cpu().numpy().round(3).tolist()}")
            print(f"action_max={action_max.detach().cpu().numpy().round(3).tolist()}")
            print(f"action_mean_abs={action_mean_abs.detach().cpu().numpy().round(3).tolist()}")
        if args_cli.save_video and frames:
            video_path = _save_episode_video(frames, recorded_video_episodes)
            if video_path is not None:
                print(f"Saved incomplete video: {video_path}")
    finally:
        pbar.close()
        env.close()


if __name__ == "__main__":
    main()  # type: ignore
    simulation_app.close()
