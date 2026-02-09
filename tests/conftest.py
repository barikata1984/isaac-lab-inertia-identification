"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def sample_joint_state():
    """Sample joint state for testing."""
    return {
        "q": np.array([0.1, -0.5, 0.3, -1.2, 0.8, 0.2]),
        "dq": np.array([0.05, -0.02, 0.01, -0.03, 0.02, 0.01]),
        "ddq": np.array([0.01, -0.005, 0.002, -0.008, 0.003, 0.001]),
    }


@pytest.fixture
def sample_wrench():
    """Sample force-torque measurement."""
    return {
        "force": np.array([1.0, 2.0, 9.81]),
        "torque": np.array([0.1, 0.2, 0.05]),
    }


@pytest.fixture
def sample_phi():
    """Sample inertial parameter vector.

    Represents a 1kg payload with:
    - CoM at (0, 0, 0.05) m
    - Diagonal inertia 0.01 kg*m^2
    """
    return np.array([
        1.0,      # m
        0.0,      # m*cx
        0.0,      # m*cy
        0.05,     # m*cz
        0.01,     # Ixx
        0.0,      # Ixy
        0.0,      # Ixz
        0.01,     # Iyy
        0.0,      # Iyz
        0.01,     # Izz
    ])


@pytest.fixture
def gravity():
    """Standard gravity vector."""
    return np.array([0.0, 0.0, -9.81])


@pytest.fixture
def kinematics():
    """PinocchioKinematics instance for UR5e."""
    try:
        from kinematics import PinocchioKinematics
        return PinocchioKinematics.for_ur5e()
    except ImportError:
        pytest.skip("kinematics package not available")
    except Exception as e:
        pytest.skip(f"Failed to load UR5e model: {e}")


@pytest.fixture
def ur5e_kin():
    """Create UR5e kinematics instance."""
    from kinematics import PinocchioKinematics
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
