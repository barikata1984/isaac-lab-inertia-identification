"""Constraint functions for excitation trajectory optimization.

All constraints are formulated as inequality constraints for
scipy.optimize.minimize (SLSQP): c(x) >= 0 means feasible.
Each function returns the minimum margin across all timesteps.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .collision_checker import CollisionChecker, CollisionConfig


@dataclass
class JointLimits:
    """Joint position, velocity, and acceleration limits for UR5e.

    Attributes:
        q_min: Lower joint position limits (6,) [rad].
        q_max: Upper joint position limits (6,) [rad].
        dq_max: Maximum joint velocities (6,) [rad/s].
        ddq_max: Maximum joint accelerations (6,) [rad/s^2].
    """

    q_min: np.ndarray = field(
        default_factory=lambda: np.full(6, -2 * np.pi)
    )
    q_max: np.ndarray = field(
        default_factory=lambda: np.full(6, 2 * np.pi)
    )
    dq_max: np.ndarray = field(
        default_factory=lambda: np.array([
            np.pi, np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi,
        ])
    )
    ddq_max: np.ndarray = field(
        default_factory=lambda: np.full(6, 8.0)
    )


@dataclass
class WorkspaceConstraintConfig:
    """Task-space displacement constraint from initial position.

    Attributes:
        max_displacement: Maximum Euclidean distance of tool0 from
            initial position [m].
    """

    max_displacement: float = 0.8


def build_trajectory_from_params(
    x: np.ndarray,
    num_joints: int,
    num_harmonics: int,
    base_freq: float,
    duration: float,
    fps: float,
    q0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct trajectory arrays from optimization variable vector.

    Args:
        x: Flat variable vector (2 * num_joints * num_harmonics,).
            Layout: [a_flat, b_flat].
        num_joints: Number of joints.
        num_harmonics: Number of Fourier harmonics.
        base_freq: Base frequency [Hz].
        duration: Trajectory duration [s].
        fps: Sampling rate [Hz].
        q0: Base joint positions (num_joints,).

    Returns:
        Tuple of (q, dq, ddq), each of shape (N_steps, num_joints).
    """
    from trajectories import WindowedFourierTrajectory, WindowedFourierTrajectoryConfig

    n = num_joints * num_harmonics
    a = x[:n].reshape(num_joints, num_harmonics)
    b = x[n:2 * n].reshape(num_joints, num_harmonics)

    cfg = WindowedFourierTrajectoryConfig(
        duration=duration,
        fps=fps,
        num_joints=num_joints,
        num_harmonics=num_harmonics,
        base_freq=base_freq,
        coefficients={"a": a.tolist(), "b": b.tolist()},
        q0=q0.tolist(),
    )
    traj = WindowedFourierTrajectory(cfg)
    q, dq, ddq = traj.get_value()

    return q, dq, ddq


class _TrajectoryCache:
    """Cache last computed trajectory to avoid recomputation.

    The objective and multiple constraint functions all call
    build_trajectory_from_params with the same x. Caching avoids
    redundant computation.
    """

    def __init__(self) -> None:
        self._last_x_hash: int | None = None
        self._last_result: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._build_params: dict | None = None

    def configure(
        self,
        num_joints: int,
        num_harmonics: int,
        base_freq: float,
        duration: float,
        fps: float,
        q0: np.ndarray,
    ) -> None:
        """Set trajectory build parameters."""
        self._build_params = {
            "num_joints": num_joints,
            "num_harmonics": num_harmonics,
            "base_freq": base_freq,
            "duration": duration,
            "fps": fps,
            "q0": q0,
        }

    def get(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get trajectory, computing only if x has changed."""
        x_hash = hash(x.tobytes())
        if x_hash != self._last_x_hash:
            self._last_result = build_trajectory_from_params(
                x, **self._build_params,
            )
            self._last_x_hash = x_hash
        return self._last_result


def make_joint_position_constraint(
    joint_limits: JointLimits,
    cache: _TrajectoryCache,
) -> Callable[[np.ndarray], float]:
    """Create joint position constraint: q_min <= q(t) <= q_max.

    Args:
        joint_limits: Joint limits.
        cache: Trajectory cache.

    Returns:
        Constraint function returning minimum margin (>= 0 means feasible).
    """
    def constraint_fn(x: np.ndarray) -> float:
        q, _, _ = cache.get(x)
        margin_lower = float(np.min(q - joint_limits.q_min))
        margin_upper = float(np.min(joint_limits.q_max - q))
        return min(margin_lower, margin_upper)

    return constraint_fn


def make_joint_velocity_constraint(
    joint_limits: JointLimits,
    cache: _TrajectoryCache,
) -> Callable[[np.ndarray], float]:
    """Create joint velocity constraint: |dq(t)| <= dq_max.

    Args:
        joint_limits: Joint limits.
        cache: Trajectory cache.

    Returns:
        Constraint function returning minimum margin.
    """
    def constraint_fn(x: np.ndarray) -> float:
        _, dq, _ = cache.get(x)
        return float(np.min(joint_limits.dq_max - np.abs(dq)))

    return constraint_fn


def make_joint_acceleration_constraint(
    joint_limits: JointLimits,
    cache: _TrajectoryCache,
) -> Callable[[np.ndarray], float]:
    """Create joint acceleration constraint: |ddq(t)| <= ddq_max.

    Args:
        joint_limits: Joint limits.
        cache: Trajectory cache.

    Returns:
        Constraint function returning minimum margin.
    """
    def constraint_fn(x: np.ndarray) -> float:
        _, _, ddq = cache.get(x)
        return float(np.min(joint_limits.ddq_max - np.abs(ddq)))

    return constraint_fn


def make_workspace_constraint(
    workspace_config: WorkspaceConstraintConfig,
    kinematics: "PinocchioKinematics",
    q0: np.ndarray,
    cache: _TrajectoryCache,
) -> Callable[[np.ndarray], float]:
    """Create task-space displacement constraint.

    Constraint: ||FK(q(t)) - FK(q0)|| <= max_displacement for all t.

    Args:
        workspace_config: Workspace constraint configuration.
        kinematics: PinocchioKinematics instance.
        q0: Initial joint configuration.
        cache: Trajectory cache.

    Returns:
        Constraint function returning minimum margin.
    """
    p0, _ = kinematics.forward_kinematics(q0)

    def constraint_fn(x: np.ndarray) -> float:
        q, _, _ = cache.get(x)
        max_disp = 0.0
        # Subsample for efficiency
        step = max(1, len(q) // 50)
        for i in range(0, len(q), step):
            p, _ = kinematics.forward_kinematics(q[i])
            disp = np.linalg.norm(p - p0)
            if disp > max_disp:
                max_disp = disp
        return workspace_config.max_displacement - max_disp

    return constraint_fn


def make_collision_constraint(
    collision_checker: CollisionChecker,
    cache: _TrajectoryCache,
    safety_margin: float = 0.005,
) -> Callable[[np.ndarray], float]:
    """Create collision avoidance constraint.

    Constraint: minimum clearance - safety_margin > 0 for all timesteps.
    The safety margin compensates for subsampling during optimization,
    ensuring full-resolution validation remains collision-free.

    Args:
        collision_checker: CollisionChecker instance.
        cache: Trajectory cache.
        safety_margin: Extra clearance margin [m] to account for
            subsampling artifacts (default: 0.005).

    Returns:
        Constraint function returning minimum clearance minus margin.
    """
    def constraint_fn(x: np.ndarray) -> float:
        q, _, _ = cache.get(x)
        return collision_checker.compute_min_clearance(q) - safety_margin

    return constraint_fn


def build_scipy_constraints(
    joint_limits: JointLimits,
    workspace_config: WorkspaceConstraintConfig,
    collision_config: CollisionConfig,
    kinematics: "PinocchioKinematics",
    collision_checker: CollisionChecker,
    q0: np.ndarray,
    cache: _TrajectoryCache,
) -> list[dict]:
    """Build list of scipy constraint dictionaries for SLSQP.

    All constraints are inequality constraints: c(x) >= 0.

    Args:
        joint_limits: Joint limits.
        workspace_config: Workspace constraint configuration.
        collision_config: Collision configuration.
        kinematics: PinocchioKinematics instance.
        collision_checker: CollisionChecker instance.
        q0: Initial joint configuration.
        cache: Trajectory cache.

    Returns:
        List of constraint dicts for scipy.optimize.minimize.
    """
    constraints = [
        {"type": "ineq", "fun": make_joint_position_constraint(
            joint_limits, cache,
        )},
        {"type": "ineq", "fun": make_joint_velocity_constraint(
            joint_limits, cache,
        )},
        {"type": "ineq", "fun": make_joint_acceleration_constraint(
            joint_limits, cache,
        )},
        {"type": "ineq", "fun": make_workspace_constraint(
            workspace_config, kinematics, q0, cache,
        )},
    ]

    if collision_config.enabled:
        constraints.append({
            "type": "ineq",
            "fun": make_collision_constraint(collision_checker, cache),
        })

    return constraints
