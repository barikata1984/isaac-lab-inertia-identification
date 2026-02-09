"""Windowed Fourier Spline Trajectory.

Combines a polynomial spline trajectory with a windowed Fourier series
to create smooth motion with oscillations in the middle portion.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base_trajectory import BaseTrajectory, BaseTrajectoryConfig
from .fourier import FourierTrajectory, FourierTrajectoryConfig
from .spline import SplineTrajectory, SplineTrajectoryConfig
from .window import WindowTrajectory, WindowTrajectoryConfig


@dataclass(kw_only=True)
class WindowedFourierSplineTrajectoryConfig(BaseTrajectoryConfig):
    """Configuration for WindowedFourierSplineTrajectory.

    Attributes:
        start_pos: Starting joint positions (required).
        end_pos: Ending joint positions (required).
        spline_type: Type of spline interpolation ("quintic" or "septic").
        num_harmonics: Number of Fourier harmonics for oscillation.
        base_freq: Base frequency for Fourier series [Hz].
        coefficients: Optional Fourier coefficients {"a": [...], "b": [...]}.
        amplitude_scale: Scale factor for random coefficient generation.
    """

    start_pos: list[float]
    end_pos: list[float]
    spline_type: str = "quintic"
    num_harmonics: int = 3
    base_freq: float = 1.0
    coefficients: dict | None = None
    amplitude_scale: float = 0.1
    target_class: str = "WindowedFourierSplineTrajectory"


class WindowedFourierSplineTrajectory(BaseTrajectory):
    """Windowed Fourier Spline Trajectory.

    Combines a polynomial spline trajectory with a windowed Fourier series
    to create smooth motion with oscillations in the middle portion.

    Mathematical formulation:
        q(t) = spline(t) + window(t) * fourier(t)

    where:
        spline(t): Polynomial trajectory (quintic/septic) connecting start to end
        window(t): 256 * r^4 * (1-r)^4 window function (0 -> 1 -> 0)
        fourier(t): Zero-mean Fourier series (oscillation component)

    Derivatives (product rule):
        dq  = ds  + dw * f   + w * df
        ddq = dds + ddw * f  + 2 * dw * df + w * ddf

    Boundary conditions guaranteed:
        - Position: Exact match at t=0 and t=T (from spline)
        - Velocity: Zero at boundaries (spline vel=0, window derivatives=0)
        - Acceleration: Zero at boundaries (spline acc=0, window derivatives=0)
    """

    def __init__(
        self,
        cfg: WindowedFourierSplineTrajectoryConfig,
        *args,
        **kwargs,
    ):
        """Initialize the trajectory generator.

        Args:
            cfg: Configuration object.
        """
        super().__init__(cfg, *args, **kwargs)

        self.cfg = cfg
        self.num_joints = len(cfg.start_pos)

        if len(cfg.end_pos) != self.num_joints:
            raise ValueError("start_pos and end_pos must have the same length.")

        # Initialize Spline Trajectory (with zero velocity/acceleration at boundaries)
        spline_cfg = SplineTrajectoryConfig(
            duration=cfg.duration,
            fps=cfg.fps,
            type=cfg.spline_type,
            start_pos=cfg.start_pos,
            end_pos=cfg.end_pos,
            start_vel=[0.0] * self.num_joints,
            end_vel=[0.0] * self.num_joints,
            start_acc=[0.0] * self.num_joints,
            end_acc=[0.0] * self.num_joints,
            start_jerk=[0.0] * self.num_joints if cfg.spline_type == "septic" else None,
            end_jerk=[0.0] * self.num_joints if cfg.spline_type == "septic" else None,
        )
        self.spline_traj = SplineTrajectory(spline_cfg)

        # Initialize Fourier Trajectory (zero-mean oscillation)
        # Scale coefficients if provided, otherwise they will be randomly generated
        scaled_coefficients = self._scale_coefficients(cfg.coefficients, cfg.amplitude_scale)

        fourier_cfg = FourierTrajectoryConfig(
            duration=cfg.duration,
            fps=cfg.fps,
            num_joints=self.num_joints,
            num_harmonics=cfg.num_harmonics,
            base_freq=cfg.base_freq,
            coefficients=scaled_coefficients,
            q0=[0.0] * self.num_joints,  # Force zero mean for oscillation part
        )
        self.fourier_traj = FourierTrajectory(fourier_cfg)

        # Initialize Window Trajectory
        window_cfg = WindowTrajectoryConfig(
            duration=cfg.duration,
            fps=cfg.fps,
            num_joints=self.num_joints,
        )
        self.window_traj = WindowTrajectory(window_cfg)

    def _scale_coefficients(
        self,
        coefficients: dict | None,
        amplitude_scale: float,
    ) -> dict | None:
        """Scale or generate Fourier coefficients.

        Args:
            coefficients: Optional dictionary with "a" and "b" arrays.
            amplitude_scale: Scale factor for random generation.

        Returns:
            Scaled coefficients dictionary or None for random generation.
        """
        if coefficients is not None:
            # Scale provided coefficients
            scaled = {}
            if "a" in coefficients:
                scaled["a"] = np.array(coefficients["a"]) * amplitude_scale
                scaled["a"] = scaled["a"].tolist()
            if "b" in coefficients:
                scaled["b"] = np.array(coefficients["b"]) * amplitude_scale
                scaled["b"] = scaled["b"].tolist()
            return scaled
        else:
            # Generate random coefficients scaled by amplitude_scale
            # The FourierTrajectory will generate random coefficients in [-1, 1]
            # We need to pre-generate and scale them
            rng = np.random.default_rng()
            a = rng.uniform(-amplitude_scale, amplitude_scale, (self.num_joints, self.cfg.num_harmonics))
            b = rng.uniform(-amplitude_scale, amplitude_scale, (self.num_joints, self.cfg.num_harmonics))
            return {"a": a.tolist(), "b": b.tolist()}

    def get_value(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate position, velocity, and acceleration at all time steps.

        Uses the product rule for derivatives:
            q   = s   + w * f
            dq  = ds  + dw * f   + w * df
            ddq = dds + ddw * f  + 2 * dw * df + w * ddf

        Returns:
            pos: (N, num_joints) Position array
            vel: (N, num_joints) Velocity array
            acc: (N, num_joints) Acceleration array
        """
        # Synchronize time arrays
        self.spline_traj.time_array = self.time_array
        self.fourier_traj.time_array = self.time_array
        self.window_traj.time_array = self.time_array

        # Get values from each component
        s, ds, dds = self.spline_traj.get_value()
        f, df, ddf = self.fourier_traj.get_value()
        w, dw, ddw = self.window_traj.get_value()

        # Combine: q = s + w * f
        q = s + w * f

        # Velocity: dq = ds + dw * f + w * df (product rule)
        dq = ds + dw * f + w * df

        # Acceleration: ddq = dds + ddw * f + 2 * dw * df + w * ddf (second derivative of product)
        ddq = dds + ddw * f + 2 * dw * df + w * ddf

        return q, dq, ddq

    def _generate(self, *args, **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate the trajectory arrays.

        Returns:
            pos: (N, num_joints) Position array
            vel: (N, num_joints) Velocity array
            acc: (N, num_joints) Acceleration array
        """
        # Adjust time array if it exceeds duration (consistent with other trajectories)
        if self.time_array[-1] > self.duration + 1e-9:
            self.time_array = self.time_array[:-1]

        pos, vel, acc = self.get_value()

        return pos, vel, acc
