#!/usr/bin/env python3
"""Visualize an optimized excitation trajectory in Isaac Sim GUI.

Loads the optimized trajectory JSON, spawns UR5e at the initial pose,
and plays back the trajectory with GUI rendering.

Usage:
    python3 visualize_trajectory.py [--trajectory PATH] [--speed FACTOR]
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Resolve symlinks so relative paths work with colcon symlink-install
_THIS_FILE = Path(os.path.realpath(__file__))


def parse_args():
    """Parse command line arguments."""
    default_traj = str(
        _THIS_FILE.parent.parent / "data" / "optimized_trajectory.json"
    )
    parser = argparse.ArgumentParser(
        description="Visualize optimized trajectory in Isaac Sim GUI"
    )
    parser.add_argument(
        "--trajectory",
        default=default_traj,
        help="Path to optimized_trajectory.json",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed factor (default: 1.0)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop trajectory playback",
    )
    return parser.parse_args()


args = parse_args()

# Isaac Sim must be initialized before other imports
os.environ["ISAAC_HEADLESS"] = "false"
from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction


def load_trajectory(json_path: str):
    """Load optimized trajectory and reconstruct via WindowedFourierTrajectory.

    Args:
        json_path: Path to optimized_trajectory.json.

    Returns:
        Tuple of (q, dq, ddq, fps, duration, q0).
    """
    pkg_src = _THIS_FILE.parent.parent / "src"
    if str(pkg_src) not in sys.path:
        sys.path.insert(0, str(pkg_src))

    from trajectories import (
        WindowedFourierTrajectory,
        WindowedFourierTrajectoryConfig,
    )

    with open(json_path) as f:
        data = json.load(f)

    cfg_data = data["config"]
    coeffs = data["coefficients"]

    cfg = WindowedFourierTrajectoryConfig(
        duration=cfg_data["duration"],
        fps=cfg_data["fps"],
        num_joints=len(cfg_data["q0"]),
        num_harmonics=cfg_data["num_harmonics"],
        base_freq=cfg_data["base_freq"],
        coefficients=coeffs,
        q0=cfg_data["q0"],
    )
    traj = WindowedFourierTrajectory(cfg)
    q, dq, ddq = traj.get_value()

    return q, dq, ddq, cfg_data["fps"], cfg_data["duration"], np.array(cfg_data["q0"])


def main():
    """Run trajectory visualization."""
    # Load trajectory
    print(f"Loading trajectory from {args.trajectory}")
    q, dq, ddq, fps, duration, q0 = load_trajectory(args.trajectory)
    n_frames = len(q)

    print(f"  Duration: {duration:.1f}s")
    print(f"  Frames: {n_frames}")
    print(f"  FPS: {fps:.0f} Hz")
    print(f"  Q0: {np.rad2deg(q0).tolist()} deg")
    print(f"  Playback speed: {args.speed}x")

    # Setup Isaac Sim world
    from ur.spawning import spawn_ur_robot

    world = World(physics_dt=1 / 240, rendering_dt=1 / 60)
    world.scene.add_default_ground_plane()

    robot = spawn_ur_robot(
        world,
        robot_type="ur5e",
        prim_path="/World/UR",
        name="ur_robot",
    )

    world.reset()

    # Set initial pose
    print("\nSetting initial pose...")
    robot.set_joint_positions(q0)
    robot.set_joint_velocities(np.zeros(6))
    for _ in range(300):
        world.step(render=True)

    print("Robot is at initial pose.")
    print("\n" + "=" * 50)
    print("Press Enter to start trajectory playback...")
    print("(Move camera / zoom in the GUI window)")
    print("=" * 50)

    # Wait for user input while keeping GUI responsive
    input_received = threading.Event()

    def wait_for_input():
        input()
        input_received.set()

    input_thread = threading.Thread(target=wait_for_input, daemon=True)
    input_thread.start()

    while not input_received.is_set():
        world.step(render=True)

    # Playback loop
    physics_dt = 1 / 240
    traj_dt = 1 / fps
    steps_per_frame = max(1, int(traj_dt / physics_dt / args.speed))

    while True:
        print(f"\nPlaying trajectory ({n_frames} frames, {duration / args.speed:.1f}s)...")
        t_start = time.time()

        for i in range(n_frames):
            robot.apply_action(ArticulationAction(joint_positions=q[i]))
            for _ in range(steps_per_frame):
                world.step(render=True)

            # Progress indicator
            if (i + 1) % (n_frames // 10) == 0:
                elapsed = time.time() - t_start
                pct = (i + 1) / n_frames * 100
                print(f"  {pct:.0f}% ({elapsed:.1f}s)")

        elapsed = time.time() - t_start
        print(f"Playback complete ({elapsed:.1f}s)")

        if not args.loop:
            break

        # Return to start
        print("Returning to initial pose...")
        robot.set_joint_positions(q0)
        robot.set_joint_velocities(np.zeros(6))
        for _ in range(200):
            world.step(render=True)

    # Keep GUI open
    print("\nTrajectory finished. Press Enter to exit...")
    input_received.clear()
    input_thread = threading.Thread(target=wait_for_input, daemon=True)
    input_thread.start()

    while not input_received.is_set():
        world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
