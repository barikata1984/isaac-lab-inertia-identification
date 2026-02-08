"""Isaac Sim inertial parameter identification using RTLS.

Spawns a UR5e robot with an attached payload in Isaac Sim,
executes an optimized excitation trajectory, and estimates
the payload's inertial parameters online using Recursive
Total Least-Squares (RTLS) with batch OLS/TLS comparison.

Reference: Kubus et al. 2008, "On-line estimation of inertial
parameters using a recursive total least-squares approach"

Usage:
    # GUI mode
    isaac-python src/iparam_identification/scripts/run_isaac_identification.py

    # Headless mode
    isaac-python src/iparam_identification/scripts/run_isaac_identification.py --headless

    # Custom noise and trajectory
    isaac-python src/iparam_identification/scripts/run_isaac_identification.py \\
        --headless --noise-force 1.0 --noise-torque 0.1 \\
        --trajectory path/to/trajectory.json --save-data results.npz
"""

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
    "--trajectory", type=str, default=None,
    help="Path to optimized trajectory JSON (default: data/optimized_trajectory.json)",
)
parser.add_argument(
    "--save-data", type=str, default=None,
    help="Path to save collected data (.npz)",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# ── Standard imports ──────────────────────────────────────────────────

import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

import omni.kit.app
from carb.eventdispatcher import get_eventdispatcher

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from pxr import UsdGeom, UsdPhysics, Gf

# ── Workspace package imports ─────────────────────────────────────────

_WS_SRC = Path("/workspace/src")


def _register_package(name: str, src_dir: Path, init: bool = True):
    """Register a workspace package for import without pip install.

    Args:
        name: Dotted module name (e.g. 'kinematics').
        src_dir: Path to the directory containing the source files.
        init: If True, execute __init__.py. Set False to skip
              (useful when __init__.py has heavy dependencies).
    """
    spec = importlib.util.spec_from_file_location(
        name,
        str(src_dir / "__init__.py"),
        submodule_search_locations=[str(src_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    if init:
        spec.loader.exec_module(mod)


# Register workspace packages
_register_package("kinematics", _WS_SRC / "kinematics" / "src")

# trajectories: skip __init__.py (imports rclpy-dependent follower_node)
# We import the specific modules we need below instead.
_register_package("trajectories", _WS_SRC / "trajectories" / "src", init=False)

# iparam_identification has subpackages that need explicit setup
_iparam_src = _WS_SRC / "iparam_identification" / "src"
_register_package("iparam_identification", _iparam_src, init=False)
# Only register subpackages needed for identification (not trajectory optimizer)
for sub in ["sensor", "estimation"]:
    _register_package(f"iparam_identification.{sub}", _iparam_src / sub)

from kinematics import PinocchioKinematics
from trajectories.windowed_fourier import (
    WindowedFourierTrajectory,
    WindowedFourierTrajectoryConfig,
)

from iparam_identification.sensor.data_types import SensorData
from iparam_identification.sensor.data_buffer import DataBuffer
from iparam_identification.sensor.contact_sensor import SimulatedForceSensor
from iparam_identification.estimation.batch_ls import BatchLeastSquares
from iparam_identification.estimation.batch_tls import BatchTotalLeastSquares
from iparam_identification.estimation.rtls import RecursiveTotalLeastSquares
from iparam_identification.estimation.base_estimator import compute_condition_number

# ── GUI close handler ─────────────────────────────────────────────────


def _force_quit(_event):
    """Terminate on window close (sim.step blocks while paused)."""
    print("\n[INFO] Window close requested. Shutting down...")
    os._exit(0)


_quit_subs = [
    get_eventdispatcher().observe_event(
        event_name=omni.kit.app.GLOBAL_EVENT_POST_QUIT,
        on_event=_force_quit,
        observer_name="identification_quit",
        order=0,
    ),
    get_eventdispatcher().observe_event(
        event_name=omni.kit.app.GLOBAL_EVENT_PRE_SHUTDOWN,
        on_event=_force_quit,
        observer_name="identification_shutdown",
        order=0,
    ),
]


# ── Constants ─────────────────────────────────────────────────────────

UR5E_URDF_PATH = (
    "/isaac-sim/exts/isaacsim.robot_motion.motion_generation"
    "/motion_policy_configs/universal_robots/ur5e/ur5e.urdf"
)

# Initial joint configuration for identification [rad]
Q0_IDENTIFICATION = np.array([
    np.pi / 2, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2
])

GRAVITY = np.array([0.0, 0.0, -9.81])


# ── Payload definition ────────────────────────────────────────────────

@dataclass
class CuboidPayload:
    """Aluminum cuboid payload specification.

    Dimensions: 10cm x 15cm x 20cm, density ~2700 kg/m^3.
    """

    width: float = 0.10    # x [m]
    height: float = 0.15   # y [m]
    depth: float = 0.20    # z [m]
    density: float = 2700.0  # kg/m^3

    # Offset from tool0 to payload COM (in tool0 frame)
    com_offset: Tuple[float, float, float] = (0.0, 0.0, 0.10)

    @property
    def volume(self) -> float:
        return self.width * self.height * self.depth

    @property
    def mass(self) -> float:
        return self.density * self.volume

    @property
    def inertia_tensor(self) -> np.ndarray:
        """Inertia tensor at COM [kg m^2]."""
        m = self.mass
        w, h, d = self.width, self.height, self.depth
        Ixx = (1 / 12) * m * (h**2 + d**2)
        Iyy = (1 / 12) * m * (w**2 + d**2)
        Izz = (1 / 12) * m * (w**2 + h**2)
        return np.diag([Ixx, Iyy, Izz])

    @property
    def inertia_at_tool0(self) -> np.ndarray:
        """Inertia tensor at tool0 (parallel axis theorem)."""
        I_com = self.inertia_tensor
        m = self.mass
        c = np.array(self.com_offset)
        return I_com + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))

    @property
    def phi_true(self) -> np.ndarray:
        """True inertial parameter vector (10,).

        phi = [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        Inertia at tool0 frame.
        """
        m = self.mass
        cx, cy, cz = self.com_offset
        I = self.inertia_at_tool0
        return np.array([
            m, m * cx, m * cy, m * cz,
            I[0, 0], I[0, 1], I[0, 2],
            I[1, 1], I[1, 2], I[2, 2],
        ])


# ── UR5e configuration ────────────────────────────────────────────────

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": float(Q0_IDENTIFICATION[0]),
            "shoulder_lift_joint": float(Q0_IDENTIFICATION[1]),
            "elbow_joint": float(Q0_IDENTIFICATION[2]),
            "wrist_1_joint": float(Q0_IDENTIFICATION[3]),
            "wrist_2_joint": float(Q0_IDENTIFICATION[4]),
            "wrist_3_joint": float(Q0_IDENTIFICATION[5]),
        },
    ),
    actuators={
        "shoulder": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_.*"],
            stiffness=800.0,
            damping=40.0,
        ),
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["elbow_joint"],
            stiffness=800.0,
            damping=40.0,
        ),
        "wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_.*"],
            stiffness=800.0,
            damping=40.0,
        ),
    },
)


# ── Scene setup ───────────────────────────────────────────────────────

def design_scene() -> tuple[dict, list[list[float]]]:
    """Create ground, light, and UR5e robot."""
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin", "Xform", translation=origins[0])

    ur5e_cfg = UR5E_CFG.copy()
    ur5e_cfg.prim_path = "/World/Origin/Robot"
    ur5e = Articulation(cfg=ur5e_cfg)

    return {"ur5e": ur5e}, origins


def create_and_attach_payload(
    payload: CuboidPayload,
    robot_prim_path: str = "/World/Origin/Robot",
    payload_prim_path: str = "/World/Origin/Payload",
) -> None:
    """Create payload cuboid and attach to tool0 via fixed joint."""
    from omni.usd import get_context
    stage = get_context().get_stage()

    # Create cube geometry
    cube = UsdGeom.Cube.Define(stage, payload_prim_path)
    scale = Gf.Vec3f(
        payload.width / 2,
        payload.height / 2,
        payload.depth / 2,
    )
    cube.AddScaleOp().Set(scale)

    # Rigid body + mass
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass_api.CreateMassAttr().Set(payload.mass)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))

    I = payload.inertia_tensor
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(float(I[0, 0]), float(I[1, 1]), float(I[2, 2]))
    )

    # Collision + visual
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetDisplayColorAttr().Set([(0.8, 0.8, 0.85)])

    # Fixed joint to tool0
    tool0_path = f"{robot_prim_path}/tool0"
    tool0_prim = stage.GetPrimAtPath(tool0_path)
    if not tool0_prim or not tool0_prim.IsValid():
        tool0_path = f"{robot_prim_path}/wrist_3_link"
        print(f"[WARN] tool0 not found, using {tool0_path}")

    joint_path = f"{payload_prim_path}/FixedJoint"
    fixed_joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    fixed_joint.CreateBody0Rel().SetTargets([tool0_path])
    fixed_joint.CreateBody1Rel().SetTargets([payload_prim_path])
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*payload.com_offset))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1, 0, 0, 0))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0, 0, 0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))

    print(f"[INFO] Payload attached to {tool0_path} "
          f"(mass={payload.mass:.2f} kg, offset={payload.com_offset})")


# ── Trajectory loading ────────────────────────────────────────────────

def load_optimized_trajectory(
    json_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load trajectory from optimized_trajectory.json.

    Returns:
        (timestamps, positions (N,6), velocities (N,6), accelerations (N,6))
    """
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


def generate_fallback_trajectory(
    duration: float = 10.0,
    dt: float = 0.01,
    n_harmonics: int = 3,
    base_freq: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a simple Fourier excitation trajectory as fallback."""
    n_steps = int(duration / dt) + 1
    t = np.linspace(0, duration, n_steps)

    q_home = Q0_IDENTIFICATION
    amplitudes = np.array([0.3, 0.15, 0.2, 0.3, 0.3, 0.3])

    rng = np.random.default_rng(42)

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

    # Smooth start/end
    blend_samples = int(1.0 / dt)
    if blend_samples > 0:
        blend = 0.5 * (1 - np.cos(np.pi * np.arange(blend_samples) / blend_samples))
        for i in range(blend_samples):
            q[i] = q_home + blend[i] * (q[i] - q_home)
            dq[i] *= blend[i]
            ddq[i] *= blend[i]
        for i in range(blend_samples):
            idx = n_steps - blend_samples + i
            factor = 0.5 * (1 + np.cos(np.pi * i / blend_samples))
            q[idx] = q_home + factor * (q[idx] - q_home)
            dq[idx] *= factor
            ddq[idx] *= factor

    return t, q, dq, ddq


# ── Result printing ───────────────────────────────────────────────────

PARAM_NAMES = ['m', 'm*cx', 'm*cy', 'm*cz',
               'Ixx', 'Ixy', 'Ixz', 'Iyy', 'Iyz', 'Izz']


def print_results(
    phi_true: np.ndarray,
    result_ols,
    result_tls,
    result_rtls,
    cond_number: float,
    n_samples: int,
    duration: float,
) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("  Inertial Parameter Identification Results")
    print("=" * 70)
    print(f"  Duration: {duration:.1f}s | Samples: {n_samples} | "
          f"Condition number: {cond_number:.2f}")

    # Method summary
    print(f"\n  {'Method':<8} | {'Mass [kg]':>10} | {'Error [%]':>10} | {'Residual':>10}")
    print(f"  {'-'*8}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for name, result in [("OLS", result_ols), ("TLS", result_tls), ("RTLS", result_rtls)]:
        mass_err = 100 * abs(result.mass - phi_true[0]) / phi_true[0]
        residual = result.residual_norm if result.residual_norm is not None else float("nan")
        print(f"  {name:<8} | {result.mass:10.4f} | {mass_err:10.2f} | {residual:10.4f}")

    # Parameter-wise detail
    print(f"\n  {'Param':<6} | {'True':>12} | {'OLS':>12} | {'TLS':>12} | {'RTLS':>12}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")

    for i, name in enumerate(PARAM_NAMES):
        tv = phi_true[i]
        ov = result_ols.phi[i]
        tlv = result_tls.phi[i]
        rv = result_rtls.phi[i]
        print(f"  {name:<6} | {tv:12.6f} | {ov:12.6f} | {tlv:12.6f} | {rv:12.6f}")

    print("=" * 70)


# ── Main identification loop ──────────────────────────────────────────

def run_identification(
    sim: SimulationContext,
    robot: Articulation,
    trajectory: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    payload: CuboidPayload,
    noise_force_std: float = 0.5,
    noise_torque_std: float = 0.05,
    save_data_path: str | None = None,
) -> dict:
    """Execute trajectory, collect data, and estimate inertial parameters.

    Returns:
        dict with estimation results for each method.
    """
    sim_dt = sim.get_physics_dt()
    timestamps, q_des, dq_des, ddq_des = trajectory
    n_steps = len(timestamps)
    traj_dt = timestamps[1] - timestamps[0] if n_steps > 1 else sim_dt

    # Physics steps per trajectory frame
    steps_per_frame = max(1, round(traj_dt / sim_dt))

    # Initialize kinematics
    print(f"[INFO] Loading kinematics from {UR5E_URDF_PATH}")
    kin = PinocchioKinematics.from_urdf_path(UR5E_URDF_PATH)

    # Initialize simulated force sensor
    sensor = SimulatedForceSensor(
        kinematics=kin,
        payload_phi=payload.phi_true,
        noise_force_std=noise_force_std,
        noise_torque_std=noise_torque_std,
        gravity=GRAVITY,
    )
    sensor.set_seed(42)

    # Initialize RTLS estimator
    rtls = RecursiveTotalLeastSquares(min_init_samples=3)

    # Data buffer for batch estimation
    data_buffer = DataBuffer(max_samples=n_steps)

    # Convergence tracking
    convergence_log = []

    # Move to start position and settle
    print("[INFO] Moving to start position...")
    start_pos = torch.tensor(
        q_des[0].astype(np.float32).reshape(1, -1), device=sim.device
    )
    robot.set_joint_position_target(start_pos)
    robot.write_data_to_sim()
    for _ in range(200):
        sim.step()
        robot.update(sim_dt)

    # Acceleration computation state
    prev_dq = None
    skip_samples = 10  # skip first samples (finite-diff not valid)

    print(f"[INFO] Running trajectory ({n_steps} frames, "
          f"{timestamps[-1]:.1f}s, {steps_per_frame} physics steps/frame)...")

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

        # Read actual state
        q = robot.data.joint_pos[0].cpu().numpy()
        dq = robot.data.joint_vel[0].cpu().numpy()

        # Finite-difference acceleration
        if prev_dq is not None:
            ddq = (dq - prev_dq) / traj_dt
        else:
            ddq = np.zeros(6)
        prev_dq = dq.copy()

        # Skip initial samples
        if i < skip_samples:
            continue

        t = timestamps[i]

        # Compute regressor
        A_k = kin.compute_regressor(q, dq, ddq, GRAVITY)

        # Simulate F/T measurement
        wrench = sensor.measure(q, dq, ddq, t)
        y_k = wrench.wrench  # (6,)

        # Online RTLS update
        phi_hat = rtls.update(A_k, y_k)

        # Store for batch estimation
        data_buffer.add_sample(SensorData(
            timestamp=t,
            q=q, dq=dq, ddq=ddq,
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
        "phi_true": payload.phi_true,
        "result_ols": result_ols,
        "result_tls": result_tls,
        "result_rtls": result_rtls,
        "condition_number": cond,
        "convergence_log": convergence_log,
    }


# ── Main ──────────────────────────────────────────────────────────────

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.3])

    # Design scene
    payload = CuboidPayload()
    print(f"[INFO] Payload: {payload.width*100:.0f}x{payload.height*100:.0f}"
          f"x{payload.depth*100:.0f} cm, mass={payload.mass:.2f} kg")

    scene_entities, scene_origins = design_scene()

    # Initialize physics
    sim.reset()

    # Attach payload after physics init
    create_and_attach_payload(payload)

    # Load trajectory
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

    timestamps = trajectory[0]
    print(f"  Duration: {timestamps[-1]:.1f}s, "
          f"Frames: {len(timestamps)}, "
          f"FPS: {1/(timestamps[1]-timestamps[0]):.0f}")

    # Run identification
    results = run_identification(
        sim=sim,
        robot=scene_entities["ur5e"],
        trajectory=trajectory,
        payload=payload,
        noise_force_std=args_cli.noise_force,
        noise_torque_std=args_cli.noise_torque,
        save_data_path=args_cli.save_data,
    )

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()
    simulation_app.close()
