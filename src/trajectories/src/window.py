from dataclasses import dataclass

import numpy as np

from .base_trajectory import BaseTrajectory, BaseTrajectoryConfig


@dataclass(kw_only=True)
class WindowTrajectoryConfig(BaseTrajectoryConfig):
    # Number of joints to match the shape of other trajectories (though window is usually same for all)
    num_joints: int  # Required, no default
    target_class: str = "WindowTrajectory"


class WindowTrajectory(BaseTrajectory):
    """
    Polynomial Window Trajectory.

    Generates a window function s(t) = 256 * r^4 * (1-r)^4 where r = t / T.
    Ensures zero value and zero derivative at boundaries.
    """

    def __init__(self, cfg: WindowTrajectoryConfig, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.num_joints = cfg.num_joints

    def get_value(self):
        """
        Calculate s, ds, dds at time t.

        Returns:
            s (num_joints,), ds (num_joints,), dds (num_joints,)
            Each is simply the scalar window value broadcasted to all joints.
        """
        T = self.duration
        if T == 0:
            zeros = np.zeros(self.num_joints)
            return zeros, zeros, zeros

        # Determine if t is scalar or array
        is_scalar = np.isscalar(self.time_array)
        t = self.time_array

        # Normalized time r
        r = t / T

        # Window s(r) = 256 * r^4 * (1-r)^4
        # Let u = r * (1-r)
        # s = 256 * u^4
        u = r * (1.0 - r)
        s = 256.0 * (u**4)

        # Derivatives
        # dr/dt = 1/T
        # du/dr = 1 - 2r
        dr_dt = 1.0 / T
        du_dr = 1.0 - 2.0 * r

        # ds/dt = ds/du * du/dt
        ds_du = 4.0 * 256.0 * (u**3)
        ds_dt = ds_du * du_dr * dr_dt

        # d2s/dt2
        # dds_dt2 = 1024/T^2 * [ 3u^2*(1-2r)^2 - 2u^3 ]
        dds_dt2 = (1024.0 / (T**2)) * (3.0 * (u**2) * (du_dr**2) - 2.0 * (u**3))

        # Broadcast to num_joints
        if is_scalar:
            q = np.full(self.num_joints, s)
            dq = np.full(self.num_joints, ds_dt)
            ddq = np.full(self.num_joints, dds_dt2)
        else:
            # (N,) -> (N, num_joints)
            q = np.tile(s[:, np.newaxis], (1, self.num_joints))
            dq = np.tile(ds_dt[:, np.newaxis], (1, self.num_joints))
            ddq = np.tile(dds_dt2[:, np.newaxis], (1, self.num_joints))

        return q, dq, ddq

    def _generate(self, *args, **kwargs):
        if self.time_array[-1] > self.duration + 1e-9:
            self.time_array = self.time_array[:-1]

        pos, vel, acc = self.get_value()

        return pos, vel, acc
