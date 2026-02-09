"""Trajectory I/O utilities for identification experiments."""

import json
from typing import Tuple

import numpy as np
from scipy.constants import g as STANDARD_GRAVITY

from models.robots.ur.ur5e import Q0_IDENTIFICATION
from trajectories.windowed_fourier import (
    WindowedFourierTrajectory,
    WindowedFourierTrajectoryConfig,
)


def load_optimized_trajectory(
    json_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load trajectory from optimized_trajectory.json.

    Returns:
        (timestamps, positions (N,6), velocities (N,6), accelerations (N,6))
    """
    with open(json_path) as f:
        data = json.load(f)

    cfg_data = data["config"]
    coeffs = data["coefficients"]

    cfg = WindowedFourierTrajectoryConfig(
        duration=cfg_data["duration"],
        fps=cfg_data["fps"],
        num_joints=len(cfg_data["q0"]),
        num_harmonics=cfg_data["num_harmonics"],
        base_freq=cfg_data["base_freq"],
        coefficients=coeffs,
        q0=cfg_data["q0"],
    )
    traj = WindowedFourierTrajectory(cfg)
    q, dq, ddq = traj.get_value()

    timestamps = np.linspace(0, cfg_data["duration"], len(q))
    return timestamps, q, dq, ddq


def generate_fallback_trajectory(
    duration: float = 10.0,
    dt: float = 0.01,
    n_harmonics: int = 3,
    base_freq: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate a simple Fourier excitation trajectory as fallback."""
    n_steps = int(duration / dt) + 1
    t = np.linspace(0, duration, n_steps)

    q_home = Q0_IDENTIFICATION
    amplitudes = np.array([0.3, 0.15, 0.2, 0.3, 0.3, 0.3])

    rng = np.random.default_rng(42)

    q = np.zeros((n_steps, 6))
    dq = np.zeros((n_steps, 6))
    ddq = np.zeros((n_steps, 6))

    for j in range(6):
        q[:, j] = q_home[j]
        for k in range(1, n_harmonics + 1):
            omega = 2 * np.pi * k * base_freq
            a_k = rng.uniform(-1, 1) * amplitudes[j] / k
            b_k = rng.uniform(-1, 1) * amplitudes[j] / k
            q[:, j] += a_k * np.sin(omega * t) + b_k * np.cos(omega * t)
            dq[:, j] += omega * (a_k * np.cos(omega * t) - b_k * np.sin(omega * t))
            ddq[:, j] += -omega**2 * (a_k * np.sin(omega * t) + b_k * np.cos(omega * t))

    # Smooth start/end
    blend_samples = int(1.0 / dt)
    if blend_samples > 0:
        blend = 0.5 * (1 - np.cos(np.pi * np.arange(blend_samples) / blend_samples))
        for i in range(blend_samples):
            q[i] = q_home + blend[i] * (q[i] - q_home)
            dq[i] *= blend[i]
            ddq[i] *= blend[i]
        for i in range(blend_samples):
            idx = n_steps - blend_samples + i
            factor = 0.5 * (1 + np.cos(np.pi * i / blend_samples))
            q[idx] = q_home + factor * (q[idx] - q_home)
            dq[idx] *= factor
            ddq[idx] *= factor

    return t, q, dq, ddq
