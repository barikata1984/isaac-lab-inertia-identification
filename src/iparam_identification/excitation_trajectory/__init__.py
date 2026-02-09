"""Trajectory optimization module for excitation trajectory design.

Provides tools for:
- Excitation trajectory optimization (condition number minimization)
- Constraint checking (joint limits, workspace, collision)
- FK-based geometric collision detection
"""

from .collision_checker import CollisionChecker, CollisionConfig
from .constraints import (
    JointLimits,
    WorkspaceConstraintConfig,
    build_trajectory_from_params,
    build_scipy_constraints,
)
from .excitation_optimizer import (
    ExcitationOptimizer,
    OptimizationResult,
    OptimizerConfig,
    compute_stacked_regressor,
    condition_number_objective,
)

__all__ = [
    "CollisionChecker",
    "CollisionConfig",
    "JointLimits",
    "WorkspaceConstraintConfig",
    "ExcitationOptimizer",
    "OptimizationResult",
    "OptimizerConfig",
    "build_trajectory_from_params",
    "build_scipy_constraints",
    "compute_stacked_regressor",
    "condition_number_objective",
]
