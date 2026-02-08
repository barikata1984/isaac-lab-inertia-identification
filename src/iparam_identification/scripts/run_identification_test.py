#!/usr/bin/env python3
"""Isaac Sim inertial parameter identification test.

This script:
1. Creates an aluminum cuboid payload attached to tool0
2. Runs the robot through an excitation trajectory
3. Collects force/torque and kinematic data
4. Estimates the inertial parameters using TLS
5. Compares with ground truth

Usage:
    ./run_identification_test.py [--headless]
"""

import argparse
import os
import sys
import threading
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np


# Create a custom print function that flushes and writes to log
_log_file = None


def log(msg: str):
    """Print message and flush, also write to log file."""
    global _log_file
    print(msg, flush=True)
    if _log_file is not None:
        _log_file.write(msg + "\n")
        _log_file.flush()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inertial parameter identification test in Isaac Sim"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run in headless mode (default: True)"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run with GUI (overrides --headless)"
    )
    return parser.parse_args()


# Parse args before importing Isaac Sim
args = parse_args()
headless = not args.gui

# Set environment variable for Isaac Sim
os.environ['ISAAC_HEADLESS'] = 'true' if headless else 'false'

# Initialize SimulationApp BEFORE any other isaacsim imports
from isaacsim.simulation_app import SimulationApp

simulation_app = SimulationApp({"headless": headless})

# Now we can import other isaacsim modules
from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
from pxr import UsdGeom, UsdPhysics, Gf, PhysxSchema, Usd


# Initial/terminal joint configuration for identification [rad]
Q0_IDENTIFICATION = np.deg2rad([90.0, -90.0, 90.0, -90.0, -90.0, 90.0])


@dataclass
class CuboidPayload:
    """Aluminum cuboid payload specification.

    Dimensions: 10cm x 15cm x 20cm
    Material: Aluminum (density ~2700 kg/m³)
    """

    # Dimensions in meters
    width: float = 0.10   # x-direction
    height: float = 0.15  # y-direction
    depth: float = 0.20   # z-direction

    # Aluminum density kg/m³
    density: float = 2700.0

    # Offset from tool0 frame to COM (in tool0 frame)
    # Assuming payload is centered and mounted below tool0
    com_offset: Tuple[float, float, float] = (0.0, 0.0, 0.10)  # 10cm below tool0

    @property
    def volume(self) -> float:
        """Volume in m³."""
        return self.width * self.height * self.depth

    @property
    def mass(self) -> float:
        """Mass in kg."""
        return self.density * self.volume

    @property
    def inertia_tensor(self) -> np.ndarray:
        """Inertia tensor at COM in kg·m²."""
        m = self.mass
        w, h, d = self.width, self.height, self.depth

        # Inertia of rectangular cuboid about COM
        Ixx = (1/12) * m * (h**2 + d**2)
        Iyy = (1/12) * m * (w**2 + d**2)
        Izz = (1/12) * m * (w**2 + h**2)

        return np.diag([Ixx, Iyy, Izz])

    @property
    def inertia_at_tool0(self) -> np.ndarray:
        """Inertia tensor at tool0 frame using parallel axis theorem."""
        I_com = self.inertia_tensor
        m = self.mass
        c = np.array(self.com_offset)

        # Parallel axis theorem: I_tool0 = I_com + m * (|c|² I - c ⊗ c)
        c_outer = np.outer(c, c)
        parallel_axis = m * (np.dot(c, c) * np.eye(3) - c_outer)

        return I_com + parallel_axis

    @property
    def phi_true(self) -> np.ndarray:
        """True inertial parameter vector.

        φ = [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]

        Note: Inertia is expressed at tool0 frame (not COM).
        """
        m = self.mass
        cx, cy, cz = self.com_offset
        I = self.inertia_at_tool0

        return np.array([
            m,
            m * cx,
            m * cy,
            m * cz,
            I[0, 0], I[0, 1], I[0, 2],
            I[1, 1], I[1, 2],
            I[2, 2],
        ])

    def __str__(self) -> str:
        I = self.inertia_at_tool0
        return (
            f"CuboidPayload:\n"
            f"  Dimensions: {self.width*100:.0f}cm x {self.height*100:.0f}cm x {self.depth*100:.0f}cm\n"
            f"  Mass: {self.mass:.4f} kg\n"
            f"  COM offset: [{self.com_offset[0]:.3f}, {self.com_offset[1]:.3f}, {self.com_offset[2]:.3f}] m\n"
            f"  Inertia at tool0:\n"
            f"    [{I[0,0]:.6f}, {I[0,1]:.6f}, {I[0,2]:.6f}]\n"
            f"    [{I[1,0]:.6f}, {I[1,1]:.6f}, {I[1,2]:.6f}]\n"
            f"    [{I[2,0]:.6f}, {I[2,1]:.6f}, {I[2,2]:.6f}]"
        )


def generate_safe_trajectory(
    duration: float = 10.0,
    dt: float = 0.01,
    n_harmonics: int = 3,
    base_freq: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a safe excitation trajectory for inertial parameter identification.

    The trajectory is designed to:
    - Stay within joint limits
    - Avoid self-collision
    - Provide sufficient excitation for parameter identification

    Args:
        duration: Total trajectory duration in seconds.
        dt: Time step in seconds.
        n_harmonics: Number of Fourier harmonics.
        base_freq: Base frequency in Hz.

    Returns:
        Tuple of (timestamps, positions, velocities, accelerations)
        - timestamps: (N,) array of time points
        - positions: (N, 6) array of joint positions
        - velocities: (N, 6) array of joint velocities
        - accelerations: (N, 6) array of joint accelerations
    """
    n_steps = int(duration / dt) + 1
    t = np.linspace(0, duration, n_steps)

    # UR5e joint limits (approximate, conservative)
    q_min = np.array([-2.0, -2.0, -1.5, -2.5, -2.0, -2.0])
    q_max = np.array([2.0, 0.5, 1.5, 0.5, 2.0, 2.0])

    # Home position (safe starting point)
    q_home = Q0_IDENTIFICATION

    # Amplitude limits (fraction of range)
    amplitudes = 0.3 * (q_max - q_min) / 2

    # Random but reproducible coefficients
    rng = np.random.default_rng(42)

    # Generate Fourier trajectory for each joint
    q = np.zeros((n_steps, 6))
    dq = np.zeros((n_steps, 6))
    ddq = np.zeros((n_steps, 6))

    for j in range(6):
        q[:, j] = q_home[j]
        for k in range(1, n_harmonics + 1):
            omega = 2 * np.pi * k * base_freq
            a_k = rng.uniform(-1, 1) * amplitudes[j] / k
            b_k = rng.uniform(-1, 1) * amplitudes[j] / k

            q[:, j] += a_k * np.sin(omega * t) + b_k * np.cos(omega * t)
            dq[:, j] += omega * (a_k * np.cos(omega * t) - b_k * np.sin(omega * t))
            ddq[:, j] += -omega**2 * (a_k * np.sin(omega * t) + b_k * np.cos(omega * t))

    # Apply smooth start/end (cosine blend)
    blend_time = 1.0  # seconds
    blend_samples = int(blend_time / dt)

    if blend_samples > 0:
        # Blend at start
        blend_factor = 0.5 * (1 - np.cos(np.pi * np.arange(blend_samples) / blend_samples))
        for i in range(blend_samples):
            q[i] = q_home + blend_factor[i] * (q[i] - q_home)
            dq[i] *= blend_factor[i]
            ddq[i] *= blend_factor[i]

        # Blend at end
        for i in range(blend_samples):
            idx = n_steps - blend_samples + i
            factor = 0.5 * (1 + np.cos(np.pi * i / blend_samples))
            q[idx] = q_home + factor * (q[idx] - q_home)
            dq[idx] *= factor
            ddq[idx] *= factor

    # Clip to joint limits
    q = np.clip(q, q_min, q_max)

    return t, q, dq, ddq


def load_optimized_trajectory(
    json_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load an optimized trajectory from JSON file.

    Args:
        json_path: Path to optimized_trajectory.json.

    Returns:
        Tuple of (timestamps, positions, velocities, accelerations).
    """
    import json
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from trajectories import WindowedFourierTrajectory, WindowedFourierTrajectoryConfig

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

    timestamps = np.linspace(0, cfg_data["duration"], len(q))
    return timestamps, q, dq, ddq


def get_stage():
    """Get the current USD stage."""
    from omni.usd import get_context
    return get_context().get_stage()


def create_payload_prim(
    payload: CuboidPayload,
    prim_path: str = "/World/Payload",
):
    """Create a payload cuboid in the USD stage.

    Args:
        payload: Payload specification.
        prim_path: USD prim path for the payload.

    Returns:
        Payload prim.
    """
    stage = get_stage()

    # Create cube geometry
    cube = UsdGeom.Cube.Define(stage, prim_path)

    # Set size (Cube in USD has size 2, so we need to scale)
    # Scale factors to get desired dimensions
    scale = Gf.Vec3f(
        payload.width / 2,
        payload.height / 2,
        payload.depth / 2
    )
    cube.AddScaleOp().Set(scale)

    # Apply rigid body physics
    rigid_api = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())

    # Set mass properties
    mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass_api.CreateMassAttr().Set(payload.mass)

    # Set center of mass (relative to prim origin)
    # Since the cube is centered at origin, COM is at origin in local frame
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))

    # Set diagonal inertia (at COM)
    I = payload.inertia_tensor
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(I[0, 0], I[1, 1], I[2, 2])
    )

    # Add collision
    collision_api = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    # Set material (visual appearance)
    # Aluminum-like color
    cube.GetDisplayColorAttr().Set([(0.8, 0.8, 0.85)])

    log(f"[INFO] Created payload at {prim_path}")
    log(f"  Mass: {payload.mass:.4f} kg")
    log(f"  Dimensions: {payload.width}m x {payload.height}m x {payload.depth}m")

    return cube.GetPrim()


def attach_payload_to_tool0(
    robot_prim_path: str,
    payload_prim_path: str,
    offset: Tuple[float, float, float],
):
    """Create a fixed joint to attach payload to tool0.

    Args:
        robot_prim_path: Path to robot prim.
        payload_prim_path: Path to payload prim.
        offset: Offset from tool0 to payload center.
    """
    stage = get_stage()

    # UR5e uses tool0 as the end-effector frame
    # Check both tool0 and wrist_3_link paths
    tool0_path = f"{robot_prim_path}/tool0"
    wrist3_path = f"{robot_prim_path}/wrist_3_link"

    # Check if tool0 exists
    tool0_prim = stage.GetPrimAtPath(tool0_path)
    if not tool0_prim or not tool0_prim.IsValid():
        log(f"[WARN] tool0 not found, using wrist_3_link")
        tool0_path = wrist3_path

    joint_path = f"{payload_prim_path}/FixedJoint"

    # Create fixed joint
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)

    # Set bodies
    fixed_joint.CreateBody0Rel().SetTargets([tool0_path])
    fixed_joint.CreateBody1Rel().SetTargets([payload_prim_path])

    # Set local poses
    # Body0 (tool0) local pose: offset in tool0 frame
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*offset))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))

    # Body1 (payload) local pose: at payload origin
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    log(f"[INFO] Attached payload to {tool0_path} with offset {offset}")


def run_simulation_and_collect_data(
    world,
    robot,
    kinematics,
    trajectory: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    headless: bool = True,
    physics_dt: float = 1/240,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Run simulation and collect data for parameter estimation.

    Args:
        world: Isaac Sim World.
        robot: Robot articulation.
        kinematics: PinocchioKinematics instance.
        trajectory: (timestamps, positions, velocities, accelerations).
        physics_dt: Physics timestep.

    Returns:
        Tuple of (A_stacked, A_list) for estimation.
    """
    timestamps, q_des, dq_des, ddq_des = trajectory
    n_steps = len(timestamps)
    dt = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 0.01

    # Data storage
    A_list = []

    # Gravity vector in base frame
    gravity = np.array([0.0, 0.0, -9.81])

    # State history for acceleration computation
    prev_dq = None
    prev_time = None

    log(f"[INFO] Running simulation ({n_steps} steps, {timestamps[-1]:.1f}s)...")

    # GUIモードの場合は入力待機（インタラクティブ）
    if not headless:
        log("\n[Waiting] Scene is ready. Press Enter to start trajectory execution...")
        log("         (You can interact with the GUI: move camera, zoom, etc.)")

        # 入力待機を別スレッドで実行
        input_received = threading.Event()

        def wait_for_input():
            """Wait for Enter key in separate thread."""
            input()
            input_received.set()

        input_thread = threading.Thread(target=wait_for_input, daemon=True)
        input_thread.start()

        # メインスレッドでアイドルレンダリング（GUIを応答的に保つ）
        while not input_received.is_set():
            world.step(render=True)

        log("\n[INFO] Starting trajectory execution...")

    # Move to start position first
    robot.set_joint_positions(q_des[0])
    robot.set_joint_velocities(np.zeros(6))

    for _ in range(100):  # Settle
        world.step(render=not headless)

    # Run trajectory
    for i in range(n_steps):
        # Set target position using ArticulationAction
        target_q = q_des[i]
        robot.apply_action(ArticulationAction(joint_positions=target_q))

        # Step simulation
        world.step(render=not headless)

        # Get actual state
        q_actual = robot.get_joint_positions()
        dq_actual = robot.get_joint_velocities()

        # Compute acceleration via finite difference
        current_time = timestamps[i]
        if prev_dq is not None and prev_time is not None:
            dt_actual = current_time - prev_time
            if dt_actual > 0:
                ddq_actual = (dq_actual - prev_dq) / dt_actual
            else:
                ddq_actual = np.zeros(6)
        else:
            ddq_actual = np.zeros(6)

        prev_dq = dq_actual.copy()
        prev_time = current_time

        # Skip first few samples (acceleration not valid)
        if i < 10:
            continue

        # Compute regressor matrix
        A_k = kinematics.compute_regressor(q_actual, dq_actual, ddq_actual, gravity)

        # For simulation, we compute y from inverse dynamics
        # In real scenario, this would come from F/T sensor
        # y = A @ phi_true (we'll compute this later with true payload)

        A_list.append(A_k)

        # Progress indicator
        if i % 100 == 0:
            log(f"  Progress: {i}/{n_steps} ({100*i/n_steps:.1f}%)")

    log(f"[INFO] Collected {len(A_list)} samples")

    # Stack data
    A_stacked = np.vstack(A_list)

    return A_stacked, A_list


def run_identification_test():
    """Run the full inertial parameter identification test."""
    global _log_file

    # Open log file
    log_path = "/tmp/iparam_identification.log"
    _log_file = open(log_path, "w")

    log("=" * 60)
    log("Inertial Parameter Identification Test")
    log("=" * 60)

    # Define payload
    payload = CuboidPayload()
    log("\n[Ground Truth]")
    log(str(payload))
    log(f"\nφ_true = {payload.phi_true}")

    # Generate or load trajectory
    log("\n[Trajectory Generation]")
    optimized_path = Path(__file__).parent.parent / "data" / "optimized_trajectory.json"
    if optimized_path.exists():
        log(f"  Loading optimized trajectory from {optimized_path}")
        trajectory = load_optimized_trajectory(str(optimized_path))
    else:
        log("  No optimized trajectory found, using default")
        trajectory = generate_safe_trajectory(
            duration=15.0,
            dt=0.01,
            n_harmonics=4,
            base_freq=0.25,
        )
    log(f"  Duration: {trajectory[0][-1]:.1f}s")
    log(f"  Samples: {len(trajectory[0])}")
    log(f"  Frequency: {1/(trajectory[0][1]-trajectory[0][0]):.1f} Hz")

    try:
        # Import ur package spawn function
        from ur.spawning import spawn_ur_robot
        from kinematics import PinocchioKinematics

        # Create world
        log("\n[Isaac Sim World Setup]")
        world = World(physics_dt=1/240, rendering_dt=1/60)

        # Setup ground plane
        world.scene.add_default_ground_plane()

        # Spawn robot
        log("\n[Robot Setup]")
        robot = spawn_ur_robot(
            world,
            robot_type="ur5e",
            prim_path="/World/UR",
            name="ur_robot",
        )

        log("\n[Payload Setup]")
        create_payload_prim(payload, "/World/Payload")

        # Attach payload to tool0
        attach_payload_to_tool0(
            robot_prim_path="/World/UR",
            payload_prim_path="/World/Payload",
            offset=payload.com_offset,
        )

        # Reset world
        world.reset()

        # Set robot to identification initial pose
        log("\n[Initial Pose]")
        log(f"  Setting q0 = {np.rad2deg(Q0_IDENTIFICATION).tolist()} deg")
        robot.set_joint_positions(Q0_IDENTIFICATION)
        robot.set_joint_velocities(np.zeros(6))
        # Settle physics
        for _ in range(200):
            world.step(render=not headless)
        log("  Robot settled at initial pose")

        # Initialize kinematics
        log("\n[Kinematics]")
        kin = PinocchioKinematics.for_ur5e()
        log("  PinocchioKinematics initialized")

        # Run simulation and collect data
        log("\n[Data Collection]")
        A_stacked, A_list = run_simulation_and_collect_data(
            world, robot, kin, trajectory, headless=headless
        )

        # Compute true y using ground truth parameters
        phi_true = payload.phi_true
        y_stacked = A_stacked @ phi_true

        # Add measurement noise (realistic sensor noise)
        noise_force_std = 0.5  # N
        noise_torque_std = 0.05  # Nm
        noise = np.zeros_like(y_stacked)
        rng = np.random.default_rng(123)
        n_samples = len(y_stacked) // 6
        for i in range(n_samples):
            noise[i*6:i*6+3] = rng.normal(0, noise_force_std, 3)
            noise[i*6+3:i*6+6] = rng.normal(0, noise_torque_std, 3)
        y_stacked += noise

        log(f"  A shape: {A_stacked.shape}")
        log(f"  y shape: {y_stacked.shape}")

        # Add package src to path for estimation import
        pkg_src = Path(__file__).parent.parent / "src"
        if str(pkg_src) not in sys.path:
            sys.path.insert(0, str(pkg_src))

        # Import estimation modules
        from estimation import (
            BatchLeastSquares,
            BatchTotalLeastSquares,
            compute_condition_number,
        )

        # Estimate parameters
        log("\n[Parameter Estimation]")

        # Condition number
        cond = compute_condition_number(A_stacked)
        log(f"  Condition number: {cond:.2f}")

        # OLS estimation
        log("\n  OLS Estimation:")
        ols = BatchLeastSquares()
        result_ols = ols.estimate(A_stacked, y_stacked)
        log(f"    φ_hat = {result_ols.phi}")
        log(f"    Mass: {result_ols.mass:.4f} kg (true: {payload.mass:.4f})")

        # TLS estimation
        log("\n  TLS Estimation:")
        tls = BatchTotalLeastSquares()
        result_tls = tls.estimate(A_stacked, y_stacked)
        log(f"    φ_hat = {result_tls.phi}")
        log(f"    Mass: {result_tls.mass:.4f} kg (true: {payload.mass:.4f})")

        # Error analysis
        log("\n[Error Analysis]")
        log("  Parameter errors (relative %):")

        param_names = ['m', 'm*cx', 'm*cy', 'm*cz',
                       'Ixx', 'Ixy', 'Ixz', 'Iyy', 'Iyz', 'Izz']

        for i, name in enumerate(param_names):
            true_val = phi_true[i]
            ols_val = result_ols.phi[i]
            tls_val = result_tls.phi[i]

            if abs(true_val) > 1e-6:
                ols_err = 100 * abs(ols_val - true_val) / abs(true_val)
                tls_err = 100 * abs(tls_val - true_val) / abs(true_val)
                log(f"    {name:6s}: OLS={ols_err:6.2f}%, TLS={tls_err:6.2f}%")
            else:
                log(f"    {name:6s}: OLS={ols_val:.6f}, TLS={tls_val:.6f} (true=0)")

        # Summary
        log("\n" + "=" * 60)
        log("Summary")
        log("=" * 60)
        log(f"True mass:      {payload.mass:.4f} kg")
        log(f"Estimated (OLS): {result_ols.mass:.4f} kg")
        log(f"Estimated (TLS): {result_tls.mass:.4f} kg")
        log(f"Mass error (OLS): {100*abs(result_ols.mass - payload.mass)/payload.mass:.2f}%")
        log(f"Mass error (TLS): {100*abs(result_tls.mass - payload.mass)/payload.mass:.2f}%")

        return {
            'payload': payload,
            'phi_true': phi_true,
            'phi_ols': result_ols.phi,
            'phi_tls': result_tls.phi,
            'mass_error_ols': abs(result_ols.mass - payload.mass) / payload.mass,
            'mass_error_tls': abs(result_tls.mass - payload.mass) / payload.mass,
        }

    except Exception as e:
        log(f"\n[ERROR] {e}")
        import traceback
        log(traceback.format_exc())
        return None

    finally:
        # Cleanup
        log("\n[Cleanup]")
        simulation_app.close()
        log("Isaac Sim closed")
        log(f"\nLog saved to: /tmp/iparam_identification.log")
        if _log_file is not None:
            _log_file.close()


def main():
    """Main entry point."""
    run_identification_test()


if __name__ == "__main__":
    main()
