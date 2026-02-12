"""Isaac Sim inertial parameter identification using RTLS.

Spawns a UR5e robot with an attached payload in Isaac Sim,
executes an optimized excitation trajectory, and estimates
the payload's inertial parameters online using Recursive
Total Least-Squares (RTLS) with batch OLS/TLS comparison.

Reference: Kubus et al. 2008, "On-line estimation of inertial
parameters using a recursive total least-squares approach"

Usage:
    # GUI mode
    isaac-python scripts/run_isaac_identification.py

    # Headless mode
    isaac-python scripts/run_isaac_identification.py --headless

    # Custom noise and trajectory
    isaac-python scripts/run_isaac_identification.py \\
        --headless --noise-force 1.0 --noise-torque 0.1 \\
        --trajectory path/to/trajectory.json --save-data results.npz
"""

# ── Preload Pinocchio's libassimp before Isaac Sim starts ─────────────
from isaac_utils.bootstrap import preload_libassimp
preload_libassimp()

# ── Isaac Lab bootstrap (must come before all other imports) ──────────

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Inertial parameter identification with UR5e in Isaac Sim"
)
parser.add_argument(
    "--noise-force", type=float, default=0.5,
    help="Force noise std dev [N] (default: 0.5)",
)
parser.add_argument(
    "--noise-torque", type=float, default=0.05,
    help="Torque noise std dev [Nm] (default: 0.05)",
)
parser.add_argument(
    "--noise-encoder-pos", type=float, default=1e-4,
    help="Joint position encoder noise std dev [rad] (default: 1e-4)",
)
parser.add_argument(
    "--noise-encoder-vel", type=float, default=1e-3,
    help="Joint velocity encoder noise std dev [rad/s] (default: 1e-3)",
)
parser.add_argument(
    "--trajectory", type=str, default=None,
    help="Path to optimized trajectory JSON (default: data/optimized_trajectory.json)",
)
parser.add_argument(
    "--save-data", type=str, default=None,
    help="Path to save collected data (.npz)",
)
parser.add_argument(
    "--playback-speed", type=float, default=1.0,
    help="GUI playback speed multiplier (default: 1.0, use 0.1 for slow-mo)",
)
parser.add_argument(
    "--payload-type", type=str, default="two-stage",
    choices=["cuboid", "cylinder", "two-stage"],
    help="Payload geometry type (default: two-stage)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# ── Standard imports ──────────────────────────────────────────────────

import time
from pathlib import Path

import numpy as np
import torch
from scipy.constants import g as STANDARD_GRAVITY

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

# ── Workspace package imports ─────────────────────────────────────────

from models.robots.ur.ur5e import URDF_PATH
from models.payloads.cuboid import CuboidPayload
from models.payloads.cylinder import CylinderPayload
from models.payloads.two_stage_cylinder import TwoStageCylinderPayload
from collision_check.capsule_checker import CapsuleCollisionChecker

from kinematics import PinocchioKinematics
from sensor.data_types import SensorData
from sensor.data_buffer import DataBuffer
from sensor.contact_sensor import SimulatedForceSensor
from iparam_identification.estimation.batch_ls import BatchLeastSquares
from iparam_identification.estimation.batch_tls import BatchTotalLeastSquares
from iparam_identification.estimation.rtls import RecursiveTotalLeastSquares
from iparam_identification.estimation.base_estimator import compute_condition_number
from iparam_identification.utils.trajectory_io import (
    load_optimized_trajectory,
    generate_fallback_trajectory,
)
from iparam_identification.utils.reporting import print_results

from isaac_utils.bootstrap import setup_quit_handler
from isaac_utils.scene import make_ur5e_cfg, design_scene, create_and_attach_payload

# ── GUI close handler ─────────────────────────────────────────────────

_quit_subs = setup_quit_handler()

# ── Gravity vector ────────────────────────────────────────────────────

GRAVITY = np.array([0.0, 0.0, -STANDARD_GRAVITY])


# ── Main identification loop ──────────────────────────────────────────

def run_identification(
    sim: SimulationContext,
    robot,
    trajectory: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    payload: CuboidPayload | CylinderPayload | TwoStageCylinderPayload,
    noise_force_std: float = 0.5,
    noise_torque_std: float = 0.05,
    noise_encoder_pos: float = 1e-4,
    noise_encoder_vel: float = 1e-3,
    save_data_path: str | None = None,
    playback_speed: float = 1.0,
) -> dict:
    """Execute trajectory, collect data, and estimate inertial parameters."""
    sim_dt = sim.get_physics_dt()
    timestamps, q_des, dq_des, ddq_des = trajectory
    n_steps = len(timestamps)
    traj_dt = timestamps[1] - timestamps[0] if n_steps > 1 else sim_dt

    # Physics steps per trajectory frame
    steps_per_frame = max(1, round(traj_dt / sim_dt))

    # Initialize kinematics
    print(f"[INFO] Loading kinematics from {URDF_PATH}")
    kin = PinocchioKinematics.from_urdf_path(URDF_PATH)

    # Initialize self-collision checker
    collision_checker = CapsuleCollisionChecker(kin.model, kin.model.createData())

    # Initialize simulated force sensor
    sensor = SimulatedForceSensor(
        kinematics=kin,
        payload_phi=payload.phi_true,
        noise_force_std=noise_force_std,
        noise_torque_std=noise_torque_std,
        gravity=GRAVITY,
    )
    sensor.set_seed(42)

    # Encoder noise RNG (separate from F/T sensor noise)
    encoder_rng = np.random.default_rng(123)
    has_encoder_noise = noise_encoder_pos > 0 or noise_encoder_vel > 0
    if has_encoder_noise:
        print(f"[INFO] Encoder noise: pos={noise_encoder_pos:.1e} rad, "
              f"vel={noise_encoder_vel:.1e} rad/s")

    # Initialize RTLS estimator
    rtls = RecursiveTotalLeastSquares(min_init_samples=3)

    # Data buffer for batch estimation
    data_buffer = DataBuffer(max_samples=n_steps)

    # Convergence tracking
    convergence_log = []

    # Acceleration computation state (from noisy velocity)
    prev_dq_meas = None
    skip_samples = 10  # skip first samples (finite-diff not valid)

    print(f"[INFO] Running trajectory ({n_steps} frames, "
          f"{timestamps[-1]:.1f}s, {steps_per_frame} physics steps/frame)...")
    if playback_speed != 1.0:
        print(f"[INFO] Playback speed: {playback_speed}x")

    wall_start = time.perf_counter()

    for i in range(n_steps):
        # Set PD target
        target = torch.tensor(
            q_des[i].astype(np.float32).reshape(1, -1), device=sim.device
        )
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()

        # Step physics
        for _ in range(steps_per_frame):
            sim.step()
        robot.update(sim_dt)

        # Throttle for playback speed
        if playback_speed > 0 and playback_speed < 1.0:
            expected_wall = timestamps[i] / playback_speed
            elapsed = time.perf_counter() - wall_start
            if elapsed < expected_wall:
                time.sleep(expected_wall - elapsed)

        # Read actual state (ground truth from simulator)
        q_true = robot.data.joint_pos[0].cpu().numpy()
        dq_true = robot.data.joint_vel[0].cpu().numpy()

        # Self-collision check (uses true state)
        collision, _min_dist, col_detail = collision_checker.check(q_true)
        if collision:
            t = timestamps[i]
            print(f"\n[NG] Trajectory aborted at t={t:.2f}s (frame {i}/{n_steps})")
            print(f"  {col_detail}")
            return {"status": "NG", "reason": col_detail, "time": t, "frame": i}

        # Simulated encoder measurement (noisy)
        if has_encoder_noise:
            q_meas = q_true + encoder_rng.normal(0, noise_encoder_pos, 6)
            dq_meas = dq_true + encoder_rng.normal(0, noise_encoder_vel, 6)
        else:
            q_meas = q_true
            dq_meas = dq_true

        # Finite-difference acceleration from noisy velocity
        if prev_dq_meas is not None:
            ddq_meas = (dq_meas - prev_dq_meas) / traj_dt
        else:
            ddq_meas = np.zeros(6)
        prev_dq_meas = dq_meas.copy()

        # Skip initial samples
        if i < skip_samples:
            continue

        t = timestamps[i]

        # Compute regressor from noisy joint measurements
        A_k = kin.compute_regressor(q_meas, dq_meas, ddq_meas, GRAVITY)

        # Simulate F/T measurement (true state + F/T sensor noise)
        wrench = sensor.measure(q_true, dq_true, ddq_meas, t)
        y_k = wrench.wrench  # (6,)

        # Online RTLS update
        phi_hat = rtls.update(A_k, y_k)

        # Store for batch estimation (noisy measurements)
        data_buffer.add_sample(SensorData(
            timestamp=t,
            q=q_meas, dq=dq_meas, ddq=ddq_meas,
            force=wrench.force, torque=wrench.torque,
            gravity=GRAVITY,
        ))

        # Log convergence
        if phi_hat is not None:
            mass_est = phi_hat[0]
            mass_err = 100 * abs(mass_est - payload.mass) / payload.mass
            convergence_log.append((t, mass_est, mass_err))

        # Progress
        if i % 200 == 0:
            mass_str = f"{phi_hat[0]:.3f}" if phi_hat is not None else "---"
            print(f"  [{i}/{n_steps}] t={t:.2f}s  mass_est={mass_str} kg")

    n_collected = len(data_buffer)
    print(f"[INFO] Collected {n_collected} samples")

    # ── Batch estimation ──────────────────────────────────────────────
    print("[INFO] Running batch estimation...")

    A_stacked, y_stacked = data_buffer.get_stacked_data(kin)
    cond = compute_condition_number(A_stacked)
    print(f"  Condition number: {cond:.2f}")

    ols = BatchLeastSquares()
    result_ols = ols.estimate(A_stacked, y_stacked)

    tls = BatchTotalLeastSquares()
    result_tls = tls.estimate(A_stacked, y_stacked)

    result_rtls = rtls.get_result()

    # Print results
    print_results(
        payload.phi_true,
        result_ols, result_tls, result_rtls,
        cond, n_collected, timestamps[-1],
    )

    # Print convergence summary
    if convergence_log:
        print(f"\n[RTLS Convergence]")
        print(f"  First estimate at t={convergence_log[0][0]:.2f}s: "
              f"mass={convergence_log[0][1]:.4f} kg "
              f"(err={convergence_log[0][2]:.2f}%)")
        print(f"  Final estimate at t={convergence_log[-1][0]:.2f}s: "
              f"mass={convergence_log[-1][1]:.4f} kg "
              f"(err={convergence_log[-1][2]:.2f}%)")

    # Save data if requested
    if save_data_path:
        data_buffer.save_to_npz(save_data_path)
        print(f"[INFO] Data saved to {save_data_path}")

    return {
        "status": "OK",
        "phi_true": payload.phi_true,
        "result_ols": result_ols,
        "result_tls": result_tls,
        "result_rtls": result_rtls,
        "condition_number": cond,
        "convergence_log": convergence_log,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    # Load trajectory first (to use its q0 as robot spawn position)
    traj_path = args_cli.trajectory
    if traj_path is None:
        default_path = (
            Path(__file__).parent.parent / "data" / "optimized_trajectory.json"
        )
        if default_path.exists():
            traj_path = str(default_path)

    if traj_path and Path(traj_path).exists():
        print(f"[INFO] Loading trajectory from {traj_path}")
        trajectory = load_optimized_trajectory(traj_path)
    else:
        print("[INFO] No optimized trajectory found, generating fallback")
        trajectory = generate_fallback_trajectory()

    timestamps, q_des = trajectory[0], trajectory[1]
    print(f"  Duration: {timestamps[-1]:.1f}s, "
          f"Frames: {len(timestamps)}, "
          f"FPS: {1/(timestamps[1]-timestamps[0]):.0f}")

    # Set up simulation
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.3])

    # Design scene (spawn robot at trajectory's initial position)
    if args_cli.payload_type == "two-stage":
        payload = TwoStageCylinderPayload()
        print(f"[INFO] Payload: Two-stage cylinder")
        print(f"  Lower: radius={payload.lower_radius*100:.0f}cm, height={payload.lower_height*100:.0f}cm, "
              f"mass={payload.lower_mass:.2f} kg")
        print(f"  Upper: radius={payload.upper_radius*100:.0f}cm, height={payload.upper_height*100:.0f}cm, "
              f"mass={payload.upper_mass:.2f} kg")
        print(f"  Total: mass={payload.mass:.2f} kg, COM at z={payload.com_offset[2]*100:.1f}cm")
    elif args_cli.payload_type == "cylinder":
        payload = CylinderPayload()
        print(f"[INFO] Payload: Cylinder (radius={payload.radius*100:.0f}cm, "
              f"height={payload.height*100:.0f}cm, mass={payload.mass:.2f} kg)")
    else:
        payload = CuboidPayload()
        print(f"[INFO] Payload: Cuboid ({payload.width*100:.0f}x{payload.height*100:.0f}"
              f"x{payload.depth*100:.0f} cm, mass={payload.mass:.2f} kg)")

    ur5e_cfg = make_ur5e_cfg(q0=q_des[0])
    scene_entities, _ = design_scene(ur5e_cfg=ur5e_cfg)

    # Embed payload as child of wrist_3_link BEFORE physics init
    # (no fixed joint — payload is part of the link's rigid body)
    create_and_attach_payload(payload)

    # Initialize physics (payload already attached)
    sim.reset()

    # Teleport robot to trajectory start position and let it settle.
    # sim.reset() spawns all joints at zero despite init_state, so we
    # must explicitly write the desired joint state to the simulation.
    robot = scene_entities["ur5e"]
    sim_dt = sim.get_physics_dt()
    start_q = torch.tensor(
        q_des[0].astype(np.float32).reshape(1, -1), device=sim.device
    )
    zero_v = torch.zeros_like(start_q)
    robot.write_joint_state_to_sim(start_q, zero_v)
    robot.set_joint_position_target(start_q)
    robot.write_data_to_sim()
    for _ in range(200):
        sim.step()
        robot.update(sim_dt)

    # Run identification
    results = run_identification(
        sim=sim,
        robot=scene_entities["ur5e"],
        trajectory=trajectory,
        payload=payload,
        noise_force_std=args_cli.noise_force,
        noise_torque_std=args_cli.noise_torque,
        noise_encoder_pos=args_cli.noise_encoder_pos,
        noise_encoder_vel=args_cli.noise_encoder_vel,
        save_data_path=args_cli.save_data,
        playback_speed=args_cli.playback_speed,
    )

    if results.get("status") == "NG":
        print(f"\n[NG] Trajectory is not feasible: {results['reason']}")
    else:
        print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
