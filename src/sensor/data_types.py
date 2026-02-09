"""Data type definitions for sensor integration.

This module defines the core data structures used throughout the
inertial parameter identification pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SensorData:
    """Single time-step sensor data for inertial parameter estimation.

    Contains all measured and derived quantities needed to construct
    the regressor matrix and observation vector.

    Attributes:
        timestamp: Time in seconds.
        q: Joint positions (6,) in radians.
        dq: Joint velocities (6,) in rad/s.
        ddq: Joint accelerations (6,) in rad/s².
        force: Measured force (3,) in sensor frame [N].
        torque: Measured torque (3,) in sensor frame [N·m].
        gravity: Gravity vector (3,) in base frame [m/s²].
    """

    timestamp: float
    q: np.ndarray
    dq: np.ndarray
    ddq: np.ndarray
    force: np.ndarray
    torque: np.ndarray
    gravity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))

    def __post_init__(self):
        """Validate and convert inputs to numpy arrays."""
        self.q = np.asarray(self.q).ravel()
        self.dq = np.asarray(self.dq).ravel()
        self.ddq = np.asarray(self.ddq).ravel()
        self.force = np.asarray(self.force).ravel()
        self.torque = np.asarray(self.torque).ravel()
        self.gravity = np.asarray(self.gravity).ravel()

        if self.q.shape != (6,):
            raise ValueError(f"q must have shape (6,), got {self.q.shape}")
        if self.dq.shape != (6,):
            raise ValueError(f"dq must have shape (6,), got {self.dq.shape}")
        if self.ddq.shape != (6,):
            raise ValueError(f"ddq must have shape (6,), got {self.ddq.shape}")
        if self.force.shape != (3,):
            raise ValueError(f"force must have shape (3,), got {self.force.shape}")
        if self.torque.shape != (3,):
            raise ValueError(f"torque must have shape (3,), got {self.torque.shape}")
        if self.gravity.shape != (3,):
            raise ValueError(f"gravity must have shape (3,), got {self.gravity.shape}")

    @property
    def wrench(self) -> np.ndarray:
        """Get combined force-torque vector (6,)."""
        return np.concatenate([self.force, self.torque])


@dataclass
class EstimationData:
    """Data prepared for parameter estimation.

    Contains the regressor matrix A and observation vector y such that:
        y = A @ φ + noise

    where φ is the 10-element inertial parameter vector.

    Attributes:
        A: Regressor matrix (6, 10).
        y: Observation vector [force; torque] (6,).
        timestamp: Optional timestamp for tracking.
    """

    A: np.ndarray
    y: np.ndarray
    timestamp: Optional[float] = None

    def __post_init__(self):
        """Validate shapes."""
        self.A = np.asarray(self.A)
        self.y = np.asarray(self.y).ravel()

        if self.A.shape != (6, 10):
            raise ValueError(f"A must have shape (6, 10), got {self.A.shape}")
        if self.y.shape != (6,):
            raise ValueError(f"y must have shape (6,), got {self.y.shape}")


@dataclass
class EstimationResult:
    """Result of inertial parameter estimation.

    Provides both raw parameter vector and structured physical quantities.

    Attributes:
        phi: Raw parameter vector (10,).
            [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
        condition_number: Condition number of the estimation problem.
        residual_norm: Norm of residuals (y - A @ phi).
        n_samples: Number of samples used in estimation.
    """

    phi: np.ndarray
    condition_number: float = np.nan
    residual_norm: float = np.nan
    n_samples: int = 0

    def __post_init__(self):
        """Validate phi shape."""
        self.phi = np.asarray(self.phi).ravel()
        if self.phi.shape != (10,):
            raise ValueError(f"phi must have shape (10,), got {self.phi.shape}")

    @property
    def mass(self) -> float:
        """Estimated mass [kg]."""
        return float(self.phi[0])

    @property
    def center_of_mass(self) -> np.ndarray:
        """Estimated center of mass [m].

        Returns (3,) array [cx, cy, cz] in sensor frame.
        Computed from phi[1:4] / phi[0].
        """
        if abs(self.phi[0]) < 1e-10:
            return np.zeros(3)
        return self.phi[1:4] / self.phi[0]

    @property
    def inertia_matrix(self) -> np.ndarray:
        """Estimated inertia matrix (3, 3) in sensor frame [kg·m²].

        Returns symmetric 3x3 matrix:
            [[Ixx, Ixy, Ixz],
             [Ixy, Iyy, Iyz],
             [Ixz, Iyz, Izz]]
        """
        Ixx, Ixy, Ixz, Iyy, Iyz, Izz = self.phi[4:10]
        return np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz],
        ])

    @property
    def inertia_at_com(self) -> np.ndarray:
        """Inertia matrix at center of mass [kg·m²].

        Applies parallel axis theorem to transform from sensor frame
        to center of mass frame.
        """
        m = self.mass
        c = self.center_of_mass
        I_sensor = self.inertia_matrix

        # Parallel axis theorem: I_com = I_sensor - m * (c^T c * I - c c^T)
        # where I is 3x3 identity
        c_outer = np.outer(c, c)
        parallel_axis = m * (np.dot(c, c) * np.eye(3) - c_outer)

        return I_sensor - parallel_axis

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "phi": self.phi.tolist(),
            "mass": self.mass,
            "center_of_mass": self.center_of_mass.tolist(),
            "inertia_matrix": self.inertia_matrix.tolist(),
            "inertia_at_com": self.inertia_at_com.tolist(),
            "condition_number": self.condition_number,
            "residual_norm": self.residual_norm,
            "n_samples": self.n_samples,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        com = self.center_of_mass
        I = self.inertia_matrix
        return (
            f"EstimationResult:\n"
            f"  Mass: {self.mass:.4f} kg\n"
            f"  Center of Mass: [{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}] m\n"
            f"  Inertia Matrix (sensor frame):\n"
            f"    [{I[0,0]:.6f}, {I[0,1]:.6f}, {I[0,2]:.6f}]\n"
            f"    [{I[1,0]:.6f}, {I[1,1]:.6f}, {I[1,2]:.6f}]\n"
            f"    [{I[2,0]:.6f}, {I[2,1]:.6f}, {I[2,2]:.6f}]\n"
            f"  Condition Number: {self.condition_number:.2f}\n"
            f"  Residual Norm: {self.residual_norm:.6f}\n"
            f"  Samples: {self.n_samples}"
        )


@dataclass
class WrenchStamped:
    """Timestamped force-torque measurement.

    Attributes:
        timestamp: Time in seconds.
        force: Force vector (3,) [N].
        torque: Torque vector (3,) [N·m].
        frame_id: Reference frame identifier.
    """

    timestamp: float
    force: np.ndarray
    torque: np.ndarray
    frame_id: str = "tool0"

    def __post_init__(self):
        """Validate and convert to numpy arrays."""
        self.force = np.asarray(self.force).ravel()
        self.torque = np.asarray(self.torque).ravel()

        if self.force.shape != (3,):
            raise ValueError(f"force must have shape (3,), got {self.force.shape}")
        if self.torque.shape != (3,):
            raise ValueError(f"torque must have shape (3,), got {self.torque.shape}")

    @property
    def wrench(self) -> np.ndarray:
        """Combined force-torque vector (6,)."""
        return np.concatenate([self.force, self.torque])
