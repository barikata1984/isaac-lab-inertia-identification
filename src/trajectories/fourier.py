from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base_trajectory import BaseTrajectory, BaseTrajectoryConfig


@dataclass(kw_only=True)
class FourierTrajectoryConfig(BaseTrajectoryConfig):
    config: Path | None = None  # Path to YAML configuration file
    num_joints: int  # Required, no default
    num_harmonics: int  # Required, no default
    base_freq: float  # Required, no default
    coefficients: dict | None = None
    q0: list[float] | None = None
    target_class: str = "FourierTrajectory"


class FourierTrajectory(BaseTrajectory):
    """
    Finite Fourier Series Trajectory.

    Represents the trajectory q_i(t) as:
    q_i(t) = q_{i,0} + sum_{k=1}^{N} ( a_{i,k} sin(2*pi*k*f*t) + b_{i,k} cos(2*pi*k*f*t) )

    Note: The paper uses rho for sine coeff and delta for cosine coeff.
    Here we use a generic name but map them consistently.
    a -> sine coefficients (rho in paper)
    b -> cosine coefficients (delta in paper)
    """

    def __init__(
        self,
        cfg: FourierTrajectoryConfig,
        *args,
        **kwargs,
    ):
        """
        Args:
            cfg: Configuration object.
        """
        super().__init__(cfg, *args, **kwargs)

        self.num_joints = cfg.num_joints
        self.num_harmonics = cfg.num_harmonics
        self.base_freq = cfg.base_freq
        self.omega_b = 2 * np.pi * cfg.base_freq

        if cfg.coefficients is None or isinstance(cfg.coefficients, str):
            # If "random" is passed or no coefficients, generate random ones for demonstration
            if cfg.coefficients == "random" or cfg.coefficients is None:
                self.a = np.random.uniform(-0.1, 0.1, (self.num_joints, self.num_harmonics))
                self.b = np.random.uniform(-0.1, 0.1, (self.num_joints, self.num_harmonics))
            else:
                self.a = np.zeros((self.num_joints, self.num_harmonics))
                self.b = np.zeros((self.num_joints, self.num_harmonics))
            self.q0 = np.array(cfg.q0) if cfg.q0 is not None else np.zeros(self.num_joints)
        else:
            self.a = np.array(cfg.coefficients.get("a", np.zeros((self.num_joints, self.num_harmonics))))
            self.b = np.array(cfg.coefficients.get("b", np.zeros((self.num_joints, self.num_harmonics))))
            if "q0" in cfg.coefficients:
                self.q0 = np.array(cfg.coefficients["q0"])
            elif cfg.q0 is not None:
                self.q0 = np.array(cfg.q0)
            else:
                self.q0 = np.zeros(self.num_joints)

    def get_value(self):
        """
        Calculate q, dq, ddq at time t.

        Returns:
            q (num_joints,), dq (num_joints,), ddq (num_joints,)
        """
        q = np.copy(self.q0)
        dq = np.zeros(self.num_joints)
        ddq = np.zeros(self.num_joints)

        # Determine if t is scalar or array
        is_scalar = np.isscalar(self.time_array)
        if not is_scalar:
            # If array, expand for consistent operations
            # Output shapes: (N, num_joints)
            t = np.array(self.time_array)
            q = np.tile(self.q0, (len(t), 1))
            dq = np.zeros((len(t), self.num_joints))
            ddq = np.zeros((len(t), self.num_joints))

        for k in range(1, self.num_harmonics + 1):
            omega_k = k * self.omega_b

            # Coefficients for k-th harmonic (0-indexed in array)
            idx = k - 1
            a_k = self.a[:, idx]  # Sine coeffs
            b_k = self.b[:, idx]  # Cosine coeffs

            wkt = omega_k * self.time_array
            sin_wkt = np.sin(wkt)
            cos_wkt = np.cos(wkt)

            if not is_scalar:
                # Reshape for broadcasting
                # a_k: (num_joints,) -> (1, num_joints)
                a_k = a_k.reshape(1, -1)
                b_k = b_k.reshape(1, -1)
                sin_wkt = sin_wkt.reshape(-1, 1)
                cos_wkt = cos_wkt.reshape(-1, 1)

            # Position
            # q += a*sin + b*cos
            q += a_k * sin_wkt + b_k * cos_wkt

            # Velocity
            # dq/dt = a*w*cos - b*w*sin
            dq += omega_k * (a_k * cos_wkt - b_k * sin_wkt)

            # Acceleration
            # d2q/dt2 = -a*w^2*sin - b*w^2*cos = -w^2 * (term)
            ddq += -(omega_k**2) * (a_k * sin_wkt + b_k * cos_wkt)

        return q, dq, ddq

    def _generate(self, *args, **kwargs):
        """
        Generate trajectory arrays.

        Returns:
            pos (N, num_joints): Position array
            vel (N, num_joints): Velocity array
            acc (N, num_joints): Acceleration array
        """
        # remove last element if it exceeds duration significantly (standard arange behavior check)
        if self.time_array[-1] > self.duration + 1e-9:
            self.time_array = self.time_array[:-1]

        pos, vel, acc = self.get_value()

        return pos, vel, acc
