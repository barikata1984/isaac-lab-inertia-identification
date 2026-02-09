from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .base_trajectory import BaseTrajectory, BaseTrajectoryConfig
from .fourier import FourierTrajectory, FourierTrajectoryConfig
from .window import WindowTrajectory, WindowTrajectoryConfig


@dataclass(kw_only=True)
class WindowedFourierTrajectoryConfig(BaseTrajectoryConfig):
    num_joints: int
    num_harmonics: int
    base_freq: float
    coefficients: dict | None = None
    q0: list[float] | None = None
    target_class: str = "WindowedFourierTrajectory"


class WindowedFourierTrajectory(BaseTrajectory):
    """
    Windowed Fourier Series Trajectory.

    Combines a Fourier Series trajectory with a Window function to ensure
    smooth start and end motions.
    
    q(t) = q0 + window(t) * fourier_oscillation(t)
    
    where:
      q0 is the base position (start/end position).
      window(t) is the polynomial window function (0 -> 1 -> 0).
      fourier_oscillation(t) is the zero-mean Fourier series part.
    """

    def __init__(
        self,
        cfg: WindowedFourierTrajectoryConfig,
        *args,
        **kwargs,
    ):
        super().__init__(cfg, *args, **kwargs)

        self.cfg = cfg
        self.num_joints = cfg.num_joints

        # Initialize Fourier Trajectory (for oscillation part)
        # We assume the Fourier part has q0=0 for pure oscillation, 
        # as we add the global q0 separately.
        fourier_cfg = FourierTrajectoryConfig(
            duration=cfg.duration,
            fps=cfg.fps,
            num_joints=cfg.num_joints,
            num_harmonics=cfg.num_harmonics,
            base_freq=cfg.base_freq,
            coefficients=cfg.coefficients,
            q0=[0.0] * cfg.num_joints, # Force zero mean for oscillation part
        )
        self.fourier_traj = FourierTrajectory(fourier_cfg)
        
        # Initialize Window Trajectory
        window_cfg = WindowTrajectoryConfig(
            duration=cfg.duration,
            fps=cfg.fps,
            num_joints=cfg.num_joints,
        )
        self.window_traj = WindowTrajectory(window_cfg)

        # Base position
        if cfg.q0 is not None:
            self.q0 = np.array(cfg.q0)
        else:
            self.q0 = np.zeros(self.num_joints)

    def get_value(self):
        """
        Calculate q, dq, ddq at time t.
        """
        # Get Fourier values (pure oscillation, centered at 0)
        # We need to temporarily set the time_array of the sub-trajectories to match self.time_array
        # in case get_value uses it (which it does in the current implementation).
        self.fourier_traj.time_array = self.time_array
        self.window_traj.time_array = self.time_array
        
        f, df, ddf = self.fourier_traj.get_value()
        w, dw, ddw = self.window_traj.get_value()

        # q = q0 + w * f
        q = self.q0 + w * f

        # dq = dw * f + w * df
        dq = dw * f + w * df

        # ddq = ddw * f + 2 * dw * df + w * ddf
        ddq = ddw * f + 2 * dw * df + w * ddf
        
        # Handle broadcasting of q0 if time_array is an array
        if not np.isscalar(self.time_array):
            # q0: (num_joints,) -> (N, num_joints)
            # w, f, etc are already (N, num_joints)
            q0_expanded = np.tile(self.q0, (len(self.time_array), 1))
            # The calculation above 'q = self.q0 + w * f' might have broadcasted correctly 
            # if w*f resulted in (N, num_joints) and q0 is (num_joints).
            # numpy broadcasting rules: (num_joints,) aligns with last dim of (N, num_joints).
            # So q0 + (w*f) works fine.
            # But let's follow the pattern of other classes if needed. 
            # Actually numpy handles (N, J) + (J,) fine.
            pass

        return q, dq, ddq

    def _generate(self, *args, **kwargs):
        if self.time_array[-1] > self.duration + 1e-9:
            self.time_array = self.time_array[:-1]

        pos, vel, acc = self.get_value()

        return pos, vel, acc
