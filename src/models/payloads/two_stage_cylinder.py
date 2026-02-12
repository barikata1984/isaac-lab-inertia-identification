"""Two-stage cylinder payload model for inertial parameter identification."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TwoStageCylinderPayload:
    """Two-stage stacked cylinder payload specification.

    Two cylinders stacked vertically:
    - Lower stage (small): attached to tool0, extends upward
    - Upper stage (large): stacked on top of lower stage

    Default: foam cylinders (polystyrene)
      Lower: radius=10cm, height=15cm
      Upper: radius=20cm, height=20cm
      Density: ~30 kg/m^3
    """

    # Lower stage (small cylinder, attached to tool0)
    lower_radius: float = 0.10    # [m]
    lower_height: float = 0.15    # [m]

    # Upper stage (large cylinder, on top of lower)
    upper_radius: float = 0.20    # [m]
    upper_height: float = 0.20    # [m]

    # Material density (same for both stages)
    density: float = 30.0  # kg/m^3 (foam/polystyrene)

    @property
    def lower_volume(self) -> float:
        """Volume of lower cylinder [m^3]."""
        return np.pi * self.lower_radius**2 * self.lower_height

    @property
    def upper_volume(self) -> float:
        """Volume of upper cylinder [m^3]."""
        return np.pi * self.upper_radius**2 * self.upper_height

    @property
    def lower_mass(self) -> float:
        """Mass of lower cylinder [kg]."""
        return self.density * self.lower_volume

    @property
    def upper_mass(self) -> float:
        """Mass of upper cylinder [kg]."""
        return self.density * self.upper_volume

    @property
    def mass(self) -> float:
        """Total mass [kg]."""
        return self.lower_mass + self.upper_mass

    def _cylinder_inertia_at_com(self, radius: float, height: float, mass: float) -> np.ndarray:
        """Inertia tensor of a cylinder at its COM.

        Args:
            radius: Cylinder radius [m]
            height: Cylinder height [m]
            mass: Cylinder mass [kg]

        Returns:
            3x3 inertia tensor at COM
        """
        Izz = (1 / 2) * mass * radius**2
        Ixx = Iyy = (1 / 12) * mass * (3 * radius**2 + height**2)
        return np.diag([Ixx, Iyy, Izz])

    @property
    def com_offset(self) -> Tuple[float, float, float]:
        """COM offset from tool0 (combined two-stage system).

        Returns:
            (x, y, z) offset in meters
        """
        # Lower cylinder COM position (z = height/2 from tool0)
        z_lower = self.lower_height / 2

        # Upper cylinder COM position (z = lower_height + height/2 from tool0)
        z_upper = self.lower_height + self.upper_height / 2

        # Combined COM (weighted average)
        m_total = self.mass
        z_com = (self.lower_mass * z_lower + self.upper_mass * z_upper) / m_total

        return (0.0, 0.0, z_com)

    @property
    def inertia_tensor(self) -> np.ndarray:
        """Inertia tensor at combined COM [kg m^2].

        Computes inertia of each cylinder at its own COM, then uses parallel
        axis theorem to transfer to the combined COM.
        """
        # Lower cylinder inertia at its own COM
        I_lower_com = self._cylinder_inertia_at_com(
            self.lower_radius, self.lower_height, self.lower_mass
        )

        # Upper cylinder inertia at its own COM
        I_upper_com = self._cylinder_inertia_at_com(
            self.upper_radius, self.upper_height, self.upper_mass
        )

        # Position of each cylinder's COM relative to combined COM
        z_com_combined = self.com_offset[2]
        z_lower_com = self.lower_height / 2
        z_upper_com = self.lower_height + self.upper_height / 2

        offset_lower = np.array([0.0, 0.0, z_lower_com - z_com_combined])
        offset_upper = np.array([0.0, 0.0, z_upper_com - z_com_combined])

        # Parallel axis theorem for each cylinder
        I_lower_at_combined = I_lower_com + self.lower_mass * (
            np.dot(offset_lower, offset_lower) * np.eye(3) - np.outer(offset_lower, offset_lower)
        )
        I_upper_at_combined = I_upper_com + self.upper_mass * (
            np.dot(offset_upper, offset_upper) * np.eye(3) - np.outer(offset_upper, offset_upper)
        )

        # Total inertia at combined COM
        return I_lower_at_combined + I_upper_at_combined

    @property
    def inertia_at_tool0(self) -> np.ndarray:
        """Inertia tensor at tool0 (parallel axis theorem from combined COM)."""
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
