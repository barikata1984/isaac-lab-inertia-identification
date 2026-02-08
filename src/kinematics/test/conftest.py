"""Pytest fixtures for kinematics tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kinematics import PinocchioKinematics


@pytest.fixture
def ur5e_kin() -> PinocchioKinematics:
    """Create UR5e kinematics instance."""
    return PinocchioKinematics.for_ur5e()


@pytest.fixture
def zero_config() -> np.ndarray:
    """Zero joint configuration."""
    return np.zeros(6)


@pytest.fixture
def random_config() -> np.ndarray:
    """Random joint configuration in [-pi, pi]."""
    np.random.seed(42)
    return np.random.uniform(-np.pi, np.pi, 6)


@pytest.fixture
def random_velocity() -> np.ndarray:
    """Random joint velocity in [-1, 1] rad/s."""
    np.random.seed(43)
    return np.random.uniform(-1.0, 1.0, 6)


@pytest.fixture
def random_acceleration() -> np.ndarray:
    """Random joint acceleration in [-0.5, 0.5] rad/s^2."""
    np.random.seed(44)
    return np.random.uniform(-0.5, 0.5, 6)
