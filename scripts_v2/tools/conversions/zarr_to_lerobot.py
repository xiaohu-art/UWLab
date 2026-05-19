# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert an OmniReset RGB-collected Zarr dataset into a LeRobot v2 dataset.

The OmniReset RGB collection script (``collect_demos.py``) writes Zarr in
diffusion-policy ReplayBuffer layout::

    rgb0.zarr/
      data/
        actions               (T, 7)
        rewards               (T,)
        dones                 (T,)
        obs/
          front_rgb           (T, 224, 224, 3)  uint8
          side_rgb            (T, 224, 224, 3)  uint8
          wrist_rgb           (T, 224, 224, 3)  uint8
          arm_joint_pos       (T, 6)
          end_effector_pose   (T, 6)
          last_arm_action     (T, 6)
          last_gripper_action (T, 1)
          ...
      meta/
        episode_ends          (N,)  int64

This script repackages it as a LeRobot v2 dataset (parquet per episode +
mp4 videos under ``videos/`` if ``--use_videos``, otherwise PNG frames).

Requires ``lerobot``:

    pip install lerobot

Usage:

    python scripts_v2/tools/conversions/zarr_to_lerobot.py \\
        --src datasets/peg/rgb0.zarr \\
        --dst datasets/peg_lerobot \\
        --repo_id local/omnireset_peg \\
        --task "peg_insertion" \\
        --fps 10

Multiple Zarr sources can be merged by passing ``--src`` multiple times.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import zarr

logger = logging.getLogger("zarr_to_lerobot")

# Image observation keys in the Zarr file -> LeRobot feature names.
DEFAULT_IMAGE_MAP = {
    "front_rgb": "observation.images.front",
    "side_rgb": "observation.images.side",
    "wrist_rgb": "observation.images.wrist",
}

# Default state composition (matches RGBPolicyCfg in data_collection_rgb_cfg.py).
# Order matters: concatenated in this order along the last axis.
DEFAULT_STATE_KEYS = [
    "last_gripper_action",  # (1,)
    "last_arm_action",      # (6,)
    "arm_joint_pos",        # (6,)
    "end_effector_pose",    # (6,)
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", action="append", required=True, help="Path to a source rgb*.zarr. Can be passed multiple times.")
    p.add_argument("--dst", required=True, type=Path, help="Output dataset root (will be created).")
    p.add_argument("--repo_id", default="local/omnireset", help="LeRobot repo_id (string label, not pushed).")
    p.add_argument("--task", default="omnireset_task", help="Task description string written to each frame.")
    p.add_argument("--fps", type=int, default=10, help="Frames per second (decimation=12 * sim_dt=1/120 -> 10 Hz).")
    p.add_argument("--state_keys", nargs="+", default=DEFAULT_STATE_KEYS,
                   help="Zarr obs keys concatenated into observation.state.")
    p.add_argument("--no_videos", action="store_true", help="Store images as PNG instead of mp4 video.")
    p.add_argument("--image_writer_threads", type=int, default=4)
    p.add_argument("--image_writer_processes", type=int, default=0)
    p.add_argument("--overwrite", action="store_true", help="Delete --dst if it exists.")
    p.add_argument("--max_episodes", type=int, default=None, help="Stop after this many episodes (debug).")
    p.add_argument("--dry_run", action="store_true", help="Verify inputs, print plan, do not write.")
    return p.parse_args()


def open_zarr(src_path: str) -> dict:
    """Open a Zarr root and return handles to all arrays we need."""
    root = zarr.open(src_path, mode="r")
    ee = root["meta/episode_ends"][:]
    if len(ee) == 0:
        raise RuntimeError(f"{src_path}: no episodes in meta/episode_ends.")
    T = int(ee[-1])

    actions = root["data/actions"]
    if actions.shape[0] != T:
        raise RuntimeError(f"{src_path}: data/actions has shape {actions.shape} but episode_ends says T={T}.")

    obs_group = root["data/obs"]
    obs_keys = list(obs_group.keys())

    return {
        "path": src_path,
        "root": root,
        "episode_ends": ee,
        "total_steps": T,
        "actions": actions,
        "obs": obs_group,
        "obs_keys": obs_keys,
    }


def validate_state_keys(handle: dict, state_keys: list[str]) -> int:
    """Ensure all requested state keys exist; return total state dim."""
    missing = [k for k in state_keys if k not in handle["obs_keys"]]
    if missing:
        raise RuntimeError(f"{handle['path']}: state keys missing in obs: {missing}. "
                           f"Available: {handle['obs_keys']}")
    dim = 0
    for k in state_keys:
        arr = handle["obs"][k]
        if arr.shape[0] != handle["total_steps"]:
            raise RuntimeError(f"{handle['path']}: obs/{k} has shape {arr.shape} but T={handle['total_steps']}.")
        if arr.ndim == 1:
            dim += 1
        else:
            dim += arr.shape[1]
    return dim


def make_features(state_dim: int, action_dim: int, image_shape: tuple[int, int, int], use_videos: bool) -> dict:
    img_dtype = "video" if use_videos else "image"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [f"s{i}" for i in range(state_dim)],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["arm_dx", "arm_dy", "arm_dz", "arm_rx", "arm_ry", "arm_rz", "gripper"][:action_dim],
        },
    }
    for _, feat_name in DEFAULT_IMAGE_MAP.items():
        features[feat_name] = {
            "dtype": img_dtype,
            "shape": image_shape,  # (H, W, C)
            "names": ["height", "width", "channels"],
        }
    return features


def build_frame(handle: dict, t: int, state_keys: list[str]) -> dict:
    """Build a single LeRobot frame dict for global step index t."""
    obs = handle["obs"]
    # state
    state_parts = []
    for k in state_keys:
        v = obs[k][t]
        v = np.atleast_1d(v).astype(np.float32)
        state_parts.append(v)
    state = np.concatenate(state_parts)
    action = handle["actions"][t].astype(np.float32)

    frame = {
        "observation.state": state,
        "action": action,
    }
    for zarr_key, feat_name in DEFAULT_IMAGE_MAP.items():
        # uint8 HWC -> LeRobot accepts numpy uint8 HWC directly
        frame[feat_name] = obs[zarr_key][t]
    return frame


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # --- open all sources, validate shapes ---
    handles = [open_zarr(s) for s in args.src]
    state_dim = validate_state_keys(handles[0], args.state_keys)
    for h in handles[1:]:
        d = validate_state_keys(h, args.state_keys)
        if d != state_dim:
            raise RuntimeError(f"state dim mismatch across sources: {state_dim} vs {d}")

    action_dim = int(handles[0]["actions"].shape[1])
    img_shape = tuple(handles[0]["obs"]["front_rgb"].shape[1:])  # (H, W, C)
    total_episodes = sum(len(h["episode_ends"]) for h in handles)
    total_steps = sum(h["total_steps"] for h in handles)

    logger.info("Sources: %d", len(handles))
    for h in handles:
        logger.info("  %s  episodes=%d  steps=%d", h["path"], len(h["episode_ends"]), h["total_steps"])
    logger.info("Episodes total: %d", total_episodes)
    logger.info("Steps total:    %d", total_steps)
    logger.info("State dim:      %d  from %s", state_dim, args.state_keys)
    logger.info("Action dim:     %d", action_dim)
    logger.info("Image shape:    %s", img_shape)
    logger.info("Storage mode:   %s", "PNG" if args.no_videos else "mp4 (video)")
    logger.info("Output root:    %s", args.dst)

    if args.dry_run:
        logger.info("--dry_run set, exiting before writing.")
        return 0

    # --- prepare output dir ---
    dst = args.dst.expanduser().resolve()
    if dst.exists():
        if not args.overwrite:
            logger.error("Output dir exists: %s  (pass --overwrite to delete)", dst)
            return 2
        import shutil
        shutil.rmtree(dst)

    # --- lazy import lerobot (only required for actual write) ---
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = make_features(state_dim, action_dim, img_shape, use_videos=not args.no_videos)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=features,
        root=str(dst),
        use_videos=not args.no_videos,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )

    # --- iterate episodes ---
    ep_global = 0
    for h in handles:
        ee = h["episode_ends"]
        starts = np.concatenate([[0], ee[:-1]])
        for ep_idx, (s, e) in enumerate(zip(starts, ee)):
            if args.max_episodes is not None and ep_global >= args.max_episodes:
                logger.info("Reached --max_episodes=%d, stopping.", args.max_episodes)
                break
            for t in range(int(s), int(e)):
                frame = build_frame(h, t, args.state_keys)
                dataset.add_frame(frame, task=args.task)
            dataset.save_episode()
            ep_global += 1
            if ep_global % 10 == 0 or ep_global == total_episodes:
                logger.info("  saved episode %d / %d  (len=%d)", ep_global, total_episodes, int(e - s))
        if args.max_episodes is not None and ep_global >= args.max_episodes:
            break

    logger.info("Done.  Wrote %d episodes to %s", ep_global, dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
