"""Excitation trajectory optimizer for inertial parameter identification.

Optimizes Fourier coefficients of WindowedFourierTrajectory to minimize
the condition number of the stacked regressor matrix, subject to joint
limits, workspace, and collision constraints.

Algorithm (following Kubus et al. 2008 Section III):
1. Generate N random initial coefficient vectors (Monte Carlo)
2. For each, run constrained optimization (SLSQP)
3. Return the solution with lowest condition number
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import minimize

from kinematics import PinocchioKinematics
from trajectories import WindowedFourierTrajectoryConfig

from .collision_checker import CollisionChecker, CollisionConfig
from .constraints import (
    JointLimits,
    WorkspaceConstraintConfig,
    _TrajectoryCache,
    build_scipy_constraints,
    build_trajectory_from_params,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Configuration for excitation trajectory optimization.

    Attributes:
        num_joints: Number of robot joints.
        num_harmonics: Number of Fourier harmonics.
        base_freq: Base frequency for Fourier series [Hz].
        duration: Trajectory duration [s].
        fps: Trajectory sampling rate [Hz].
        q0: Initial/terminal joint configuration [rad].
        joint_limits: Joint position/velocity/acceleration limits.
        collision: Collision checking configuration.
        workspace: Task-space constraint configuration.
        subsample_factor: Evaluate every N-th timestep in objective.
        n_monte_carlo: Number of Monte Carlo random initializations.
        max_iter_per_start: Maximum iterations per optimization start.
        optimizer_method: scipy.optimize method name.
        ftol: Function tolerance for convergence.
        seed: Random seed for reproducibility.
    """

    num_joints: int = 6
    num_harmonics: int = 5
    base_freq: float = 0.1
    duration: float = 10.0
    fps: float = 100.0
    q0: np.ndarray = field(default_factory=lambda: np.array([
        np.pi / 2, -np.pi / 2, np.pi / 2,
        -np.pi / 2, -np.pi / 2, np.pi / 2,
    ]))
    joint_limits: JointLimits = field(default_factory=JointLimits)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    workspace: WorkspaceConstraintConfig = field(
        default_factory=WorkspaceConstraintConfig,
    )
    subsample_factor: int = 10
    n_monte_carlo: int = 20
    max_iter_per_start: int = 200
    optimizer_method: str = "SLSQP"
    ftol: float = 1e-6
    seed: int = 42


@dataclass
class OptimizationResult:
    """Result of excitation trajectory optimization.

    Attributes:
        x_opt: Optimal Fourier coefficient vector.
        condition_number: Final condition number.
        a_opt: Optimal sine coefficients (num_joints, num_harmonics).
        b_opt: Optimal cosine coefficients (num_joints, num_harmonics).
        q0: Base joint positions (num_joints,).
        trajectory_config: Config to reconstruct the optimal trajectory.
        n_evaluations: Total objective evaluations.
        wall_time: Total optimization wall time [s].
        n_restarts: Number of Monte Carlo restarts performed.
        best_start_index: Which restart produced the best result.
    """

    x_opt: np.ndarray
    condition_number: float
    a_opt: np.ndarray
    b_opt: np.ndarray
    q0: np.ndarray
    trajectory_config: WindowedFourierTrajectoryConfig
    n_evaluations: int = 0
    wall_time: float = 0.0
    n_restarts: int = 0
    best_start_index: int = 0


def compute_stacked_regressor(
    kinematics: PinocchioKinematics,
    q: np.ndarray,
    dq: np.ndarray,
    ddq: np.ndarray,
    subsample_factor: int = 1,
    gravity: np.ndarray | None = None,
) -> np.ndarray:
    """Compute stacked regressor matrix for a trajectory.

    Args:
        kinematics: PinocchioKinematics instance.
        q: Joint positions (N, 6).
        dq: Joint velocities (N, 6).
        ddq: Joint accelerations (N, 6).
        subsample_factor: Use every N-th sample.
        gravity: Gravity vector (3,). Default [0, 0, -9.81].

    Returns:
        Stacked regressor matrix (M*6, 10) where M = N // subsample_factor.
    """
    if gravity is None:
        gravity = np.array([0.0, 0.0, -9.81])

    A_list = []
    for i in range(0, len(q), subsample_factor):
        A_k = kinematics.compute_regressor(q[i], dq[i], ddq[i], gravity)
        A_list.append(A_k)

    return np.vstack(A_list)


def condition_number_objective(
    x: np.ndarray,
    kinematics: PinocchioKinematics,
    cache: _TrajectoryCache,
    subsample_factor: int = 1,
) -> float:
    """Compute condition number of the stacked regressor matrix.

    This is the objective function to be minimized.

    Args:
        x: Optimization variable vector.
        kinematics: PinocchioKinematics instance.
        cache: Trajectory cache for efficiency.
        subsample_factor: Subsampling factor.

    Returns:
        Condition number. Returns 1e12 on failure (SLSQP needs finite).
    """
    try:
        q, dq, ddq = cache.get(x)
        A = compute_stacked_regressor(
            kinematics, q, dq, ddq, subsample_factor,
        )
        sv = np.linalg.svd(A, compute_uv=False)

        if sv[-1] < 1e-15:
            return 1e12

        return float(sv[0] / sv[-1])
    except Exception:
        return 1e12


class ExcitationOptimizer:
    """Excitation trajectory optimizer.

    Optimizes Fourier coefficients of WindowedFourierTrajectory to
    minimize the condition number of the stacked regressor matrix.

    Usage:
        kinematics = PinocchioKinematics.for_ur5e()
        config = OptimizerConfig(num_harmonics=5, base_freq=0.1)
        optimizer = ExcitationOptimizer(config, kinematics)
        result = optimizer.optimize()
    """

    def __init__(
        self,
        config: OptimizerConfig,
        kinematics: PinocchioKinematics,
    ):
        """Initialize optimizer.

        Args:
            config: Optimization configuration.
            kinematics: PinocchioKinematics instance.
        """
        self.config = config
        self.kinematics = kinematics

        self.collision_checker = CollisionChecker(
            model=kinematics.model,
            config=config.collision,
        )

        # Number of optimization variables
        self.n_vars = 2 * config.num_joints * config.num_harmonics

        # Shared trajectory cache
        self._cache = _TrajectoryCache()
        self._cache.configure(
            num_joints=config.num_joints,
            num_harmonics=config.num_harmonics,
            base_freq=config.base_freq,
            duration=config.duration,
            fps=config.fps,
            q0=config.q0,
        )

        self._n_evals = 0

    def _generate_random_x0(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a random initial coefficient vector.

        Amplitude scales inversely with harmonic number for smooth
        initial trajectories.

        Args:
            rng: NumPy random generator.

        Returns:
            Random coefficient vector (n_vars,).
        """
        n_j = self.config.num_joints
        n_h = self.config.num_harmonics

        scale = np.zeros((n_j, n_h))
        for k in range(n_h):
            scale[:, k] = 0.3 / (k + 1)

        a = rng.uniform(-1, 1, (n_j, n_h)) * scale
        b = rng.uniform(-1, 1, (n_j, n_h)) * scale

        return np.concatenate([a.ravel(), b.ravel()])

    def _objective(self, x: np.ndarray) -> float:
        """Objective function with evaluation counting.

        Args:
            x: Optimization variable vector.

        Returns:
            Condition number.
        """
        self._n_evals += 1
        return condition_number_objective(
            x, self.kinematics, self._cache, self.config.subsample_factor,
        )

    def _run_single_optimization(
        self,
        x0: np.ndarray,
        constraints: list[dict],
    ) -> tuple[np.ndarray, float, bool]:
        """Run a single SLSQP optimization from given starting point.

        Args:
            x0: Initial point.
            constraints: Scipy constraint list.

        Returns:
            Tuple of (x_opt, f_opt, success).
        """
        result = minimize(
            self._objective,
            x0,
            method=self.config.optimizer_method,
            constraints=constraints,
            options={
                "maxiter": self.config.max_iter_per_start,
                "ftol": self.config.ftol,
                "disp": False,
            },
        )

        return result.x, float(result.fun), result.success

    def optimize(self, verbose: bool = True) -> OptimizationResult:
        """Run the full multi-start optimization.

        Args:
            verbose: Whether to log progress.

        Returns:
            OptimizationResult with optimal trajectory parameters.
        """
        rng = np.random.default_rng(self.config.seed)

        constraints = build_scipy_constraints(
            joint_limits=self.config.joint_limits,
            workspace_config=self.config.workspace,
            collision_config=self.config.collision,
            kinematics=self.kinematics,
            collision_checker=self.collision_checker,
            q0=self.config.q0,
            cache=self._cache,
        )

        best_x: Optional[np.ndarray] = None
        best_f = np.inf
        best_idx = -1

        t_start = time.time()
        self._n_evals = 0

        for i in range(self.config.n_monte_carlo):
            x0 = self._generate_random_x0(rng)

            # Evaluate initial condition number
            kappa_init = self._objective(x0)
            if verbose:
                logger.info(
                    "  Start %d/%d: initial kappa = %.2f",
                    i + 1, self.config.n_monte_carlo, kappa_init,
                )

            x_opt, f_opt, success = self._run_single_optimization(
                x0, constraints,
            )

            if verbose:
                status = "converged" if success else "not converged"
                logger.info(
                    "  Start %d/%d: kappa = %.2f (%s)",
                    i + 1, self.config.n_monte_carlo, f_opt, status,
                )

            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()
                best_idx = i

        wall_time = time.time() - t_start

        if best_x is None:
            raise RuntimeError("All optimization starts failed")

        # Extract optimal coefficients
        n = self.config.num_joints * self.config.num_harmonics
        a_opt = best_x[:n].reshape(
            self.config.num_joints, self.config.num_harmonics,
        )
        b_opt = best_x[n:].reshape(
            self.config.num_joints, self.config.num_harmonics,
        )

        trajectory_config = WindowedFourierTrajectoryConfig(
            duration=self.config.duration,
            fps=self.config.fps,
            num_joints=self.config.num_joints,
            num_harmonics=self.config.num_harmonics,
            base_freq=self.config.base_freq,
            coefficients={"a": a_opt.tolist(), "b": b_opt.tolist()},
            q0=self.config.q0.tolist(),
        )

        result = OptimizationResult(
            x_opt=best_x,
            condition_number=best_f,
            a_opt=a_opt,
            b_opt=b_opt,
            q0=self.config.q0,
            trajectory_config=trajectory_config,
            n_evaluations=self._n_evals,
            wall_time=wall_time,
            n_restarts=self.config.n_monte_carlo,
            best_start_index=best_idx,
        )

        if verbose:
            logger.info("Optimization complete:")
            logger.info("  Best kappa: %.2f", best_f)
            logger.info("  Best start: %d", best_idx + 1)
            logger.info("  Total evaluations: %d", self._n_evals)
            logger.info("  Wall time: %.1fs", wall_time)

        return result

    def validate_trajectory(
        self,
        result: OptimizationResult,
    ) -> dict:
        """Validate the optimized trajectory at full resolution.

        Performs detailed checks without subsampling.

        Args:
            result: Optimization result to validate.

        Returns:
            Dictionary with validation metrics.
        """
        q, dq, ddq = build_trajectory_from_params(
            result.x_opt,
            self.config.num_joints,
            self.config.num_harmonics,
            self.config.base_freq,
            self.config.duration,
            self.config.fps,
            self.config.q0,
        )

        # Full-resolution condition number
        A_full = compute_stacked_regressor(
            self.kinematics, q, dq, ddq, subsample_factor=1,
        )
        sv = np.linalg.svd(A_full, compute_uv=False)
        kappa_full = float(sv[0] / sv[-1]) if sv[-1] > 1e-15 else np.inf

        # Joint limit margins
        q_margin_lower = float(np.min(q - self.config.joint_limits.q_min))
        q_margin_upper = float(np.min(self.config.joint_limits.q_max - q))
        dq_margin = float(np.min(self.config.joint_limits.dq_max - np.abs(dq)))
        ddq_margin = float(np.min(
            self.config.joint_limits.ddq_max - np.abs(ddq)
        ))

        # Workspace check
        p0, _ = self.kinematics.forward_kinematics(self.config.q0)
        max_disp = 0.0
        for i in range(len(q)):
            p, _ = self.kinematics.forward_kinematics(q[i])
            max_disp = max(max_disp, float(np.linalg.norm(p - p0)))

        # Collision check (full resolution)
        min_clearance = self.collision_checker.compute_min_clearance(
            q, subsample_factor=1,
        )

        # Allow small numerical tolerance for SLSQP boundary solutions
        tol = 1e-2
        all_satisfied = all([
            q_margin_lower >= -tol,
            q_margin_upper >= -tol,
            dq_margin >= -tol,
            ddq_margin >= -tol,
            max_disp <= self.config.workspace.max_displacement + tol,
            min_clearance >= -tol,
        ])

        return {
            "condition_number_full": kappa_full,
            "condition_number_subsampled": result.condition_number,
            "n_timesteps": len(q),
            "q_margin_lower": q_margin_lower,
            "q_margin_upper": q_margin_upper,
            "dq_margin": dq_margin,
            "ddq_margin": ddq_margin,
            "max_tool0_displacement": max_disp,
            "min_collision_clearance": min_clearance,
            "all_constraints_satisfied": all_satisfied,
            "singular_values": sv.tolist(),
        }
