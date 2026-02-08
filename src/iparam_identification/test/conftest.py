"""Pytest configuration and fixtures for iparam_identification tests."""

import sys
from pathlib import Path
import types

# Setup package path before any imports
_package_root = Path(__file__).parent.parent.absolute()
_src_dir = _package_root / "src"


def _create_package(name: str, path: Path, parent=None):
    """Create a module/package and register it in sys.modules."""
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(path)]
    pkg.__file__ = str(path / "__init__.py")
    pkg.__package__ = name
    sys.modules[name] = pkg

    if parent is not None:
        setattr(parent, name.split('.')[-1], pkg)

    return pkg


def _exec_init(pkg, path: Path):
    """Execute __init__.py for a package."""
    init_file = path / "__init__.py"
    if init_file.exists():
        code = compile(init_file.read_text(), str(init_file), 'exec')
        exec(code, pkg.__dict__)


def _setup_iparam_identification():
    """Setup the iparam_identification package for testing."""
    pkg_name = "iparam_identification"

    if pkg_name in sys.modules:
        return sys.modules[pkg_name]

    # Create main package
    pkg = _create_package(pkg_name, _src_dir)

    # Create and setup subpackages
    for subpkg_name in ["sensor", "estimation", "trajectory"]:
        subpkg_dir = _src_dir / subpkg_name
        if not subpkg_dir.exists():
            continue

        full_name = f"{pkg_name}.{subpkg_name}"
        subpkg = _create_package(full_name, subpkg_dir, parent=pkg)

        # Execute subpackage __init__.py
        _exec_init(subpkg, subpkg_dir)

    # Execute main package __init__.py last (after subpackages are set up)
    _exec_init(pkg, _src_dir)

    return pkg


# Setup package
_setup_iparam_identification()


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
    - Diagonal inertia 0.01 kg·m²
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
    """PinocchioKinematics instance for UR5e.

    This fixture requires kinematics package and ur_description to be available.
    """
    try:
        from kinematics import PinocchioKinematics
        return PinocchioKinematics.for_ur5e()
    except ImportError:
        pytest.skip("kinematics package not available")
    except Exception as e:
        pytest.skip(f"Failed to load UR5e model: {e}")
