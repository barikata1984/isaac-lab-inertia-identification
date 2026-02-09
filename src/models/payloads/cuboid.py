"""Cuboid payload model for inertial parameter identification."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CuboidPayload:
    """Cuboid payload specification.

    Default: aluminum cuboid 10cm x 15cm x 20cm, density ~2700 kg/m^3.
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
