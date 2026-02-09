"""Trajectories package - unified trajectory generation."""

from .base_trajectory import BaseTrajectory, BaseTrajectoryConfig
from .fourier import FourierTrajectory, FourierTrajectoryConfig
from .spline import SplineTrajectory, SplineTrajectoryConfig
from .window import WindowTrajectory, WindowTrajectoryConfig
from .windowed_fourier import (
    WindowedFourierTrajectory,
    WindowedFourierTrajectoryConfig,
)
from .windowed_fourier_spline import (
    WindowedFourierSplineTrajectory,
    WindowedFourierSplineTrajectoryConfig,
)

__all__ = [
    "BaseTrajectory",
    "BaseTrajectoryConfig",
    "FourierTrajectory",
    "FourierTrajectoryConfig",
    "SplineTrajectory",
    "SplineTrajectoryConfig",
    "WindowTrajectory",
    "WindowTrajectoryConfig",
    "WindowedFourierTrajectory",
    "WindowedFourierTrajectoryConfig",
    "WindowedFourierSplineTrajectory",
    "WindowedFourierSplineTrajectoryConfig",
]
