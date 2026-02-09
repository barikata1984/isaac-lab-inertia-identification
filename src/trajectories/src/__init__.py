"""Trajectories package - unified trajectory generation and playback."""

# Trajectory generation
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

# Trajectory playback (requires ROS2)
try:
    from .follower_node import main as follower_main
except ImportError:
    follower_main = None

__all__ = [
    # Generation
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
    # Playback
    "follower_main",
]

