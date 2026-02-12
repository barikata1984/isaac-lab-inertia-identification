"""Cylinder payload model for inertial parameter identification."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CylinderPayload:
    """Cylinder payload specification.

    The cylinder is attached via its bottom face to the robot's tool0.
    The cylinder extends upward (positive Z direction) from tool0.
    Default: foam cylinder (polystyrene), radius=20cm, height=35cm, density ~30 kg/m^3.
    """

    radius: float = 0.20    # [m]
    height: float = 0.35    # [m]
    density: float = 30.0  # kg/m^3 (foam/polystyrene)

    # Offset from tool0 to payload COM (in tool0 frame)
    # For bottom-face attachment, COM is at +height/2 along z-axis (upward)
    @property
    def com_offset(self) -> Tuple[float, float, float]:
        """COM offset from tool0 (bottom face attachment, cylinder extends upward)."""
        return (0.0, 0.0, self.height / 2)

    @property
    def volume(self) -> float:
        """Volume [m^3]."""
        return np.pi * self.radius**2 * self.height

    @property
    def mass(self) -> float:
        """Mass [kg]."""
        return self.density * self.volume

    @property
    def inertia_tensor(self) -> np.ndarray:
        """Inertia tensor at COM [kg m^2].

        For a solid cylinder of radius r and height h:
        - Izz (along axis): (1/2) * m * r^2
        - Ixx, Iyy (perpendicular): (1/12) * m * (3*r^2 + h^2)
        """
        m = self.mass
        r = self.radius
        h = self.height
        Izz = (1 / 2) * m * r**2
        Ixx = Iyy = (1 / 12) * m * (3 * r**2 + h**2)
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
