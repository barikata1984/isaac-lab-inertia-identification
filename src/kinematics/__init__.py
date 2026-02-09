"""Kinematics package - Pinocchio-based kinematics computation.

Provides forward kinematics, velocity, and acceleration for UR robots
using Pinocchio library and ur_description URDF.

Output convention: [linear, angular] (v, omega) for velocity/acceleration.
"""

from .kinematics import (
    Tool0State,
    PinocchioKinematics,
    load_ur_model,
    compute_regressor_matrix,
)

__all__ = [
    "Tool0State",
    "PinocchioKinematics",
    "load_ur_model",
    "compute_regressor_matrix",
]
