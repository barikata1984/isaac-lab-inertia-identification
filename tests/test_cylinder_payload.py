"""Tests for cylinder payload model."""

import numpy as np
import pytest

from models.payloads.cylinder import CylinderPayload


def test_cylinder_defaults():
    """Test default cylinder payload parameters."""
    payload = CylinderPayload()
    assert payload.radius == 0.20
    assert payload.height == 0.35
    assert payload.density == 30.0  # Foam/polystyrene


def test_cylinder_volume():
    """Test volume calculation."""
    payload = CylinderPayload(radius=0.1, height=0.2)
    expected = np.pi * 0.1**2 * 0.2
    assert np.isclose(payload.volume, expected)


def test_cylinder_mass():
    """Test mass calculation."""
    payload = CylinderPayload(radius=0.1, height=0.2, density=1000.0)
    expected_volume = np.pi * 0.1**2 * 0.2
    expected_mass = 1000.0 * expected_volume
    assert np.isclose(payload.mass, expected_mass)


def test_cylinder_com_offset():
    """Test COM offset for bottom-face attachment."""
    payload = CylinderPayload(height=0.4)
    # Bottom face at tool0, COM at +height/2 (upward)
    assert payload.com_offset == (0.0, 0.0, 0.2)


def test_cylinder_inertia_tensor():
    """Test inertia tensor at COM."""
    payload = CylinderPayload(radius=0.1, height=0.2, density=2700.0)
    m = payload.mass
    r = 0.1
    h = 0.2

    # Expected inertia (solid cylinder)
    Izz_expected = (1 / 2) * m * r**2
    Ixx_expected = Iyy_expected = (1 / 12) * m * (3 * r**2 + h**2)

    I = payload.inertia_tensor
    assert I.shape == (3, 3)
    assert np.isclose(I[0, 0], Ixx_expected)
    assert np.isclose(I[1, 1], Iyy_expected)
    assert np.isclose(I[2, 2], Izz_expected)
    # Off-diagonal should be zero (diagonal tensor)
    assert np.allclose(I - np.diag(np.diag(I)), 0)


def test_cylinder_inertia_at_tool0():
    """Test parallel axis theorem for inertia at tool0."""
    payload = CylinderPayload(radius=0.1, height=0.2, density=2700.0)
    I_com = payload.inertia_tensor
    I_tool0 = payload.inertia_at_tool0

    # Parallel axis theorem: I = I_com + m*(c·c*I - c⊗c)
    m = payload.mass
    c = np.array(payload.com_offset)
    c_dot_c = np.dot(c, c)
    I_expected = I_com + m * (c_dot_c * np.eye(3) - np.outer(c, c))

    assert np.allclose(I_tool0, I_expected)


def test_cylinder_phi_true():
    """Test 10-parameter inertial vector."""
    payload = CylinderPayload(radius=0.1, height=0.2, density=2700.0)
    phi = payload.phi_true

    assert phi.shape == (10,)

    # Check components
    m = payload.mass
    cx, cy, cz = payload.com_offset
    I = payload.inertia_at_tool0

    assert np.isclose(phi[0], m)
    assert np.isclose(phi[1], m * cx)
    assert np.isclose(phi[2], m * cy)
    assert np.isclose(phi[3], m * cz)
    assert np.isclose(phi[4], I[0, 0])
    assert np.isclose(phi[5], I[0, 1])
    assert np.isclose(phi[6], I[0, 2])
    assert np.isclose(phi[7], I[1, 1])
    assert np.isclose(phi[8], I[1, 2])
    assert np.isclose(phi[9], I[2, 2])


def test_cylinder_phi_shape():
    """Ensure phi_true is always (10,) for different cylinder dimensions."""
    payloads = [
        CylinderPayload(radius=0.1, height=0.3),
        CylinderPayload(radius=0.2, height=0.35),
        CylinderPayload(radius=0.15, height=0.4, density=7800.0),  # Steel
    ]
    for payload in payloads:
        assert payload.phi_true.shape == (10,)


def test_cylinder_mass_positive():
    """Ensure mass is always positive."""
    payload = CylinderPayload(radius=0.1, height=0.2, density=2700.0)
    assert payload.mass > 0


def test_cylinder_inertia_positive_definite():
    """Ensure inertia tensor is positive definite."""
    payload = CylinderPayload(radius=0.1, height=0.2, density=2700.0)
    I = payload.inertia_tensor
    eigenvalues = np.linalg.eigvalsh(I)
    assert np.all(eigenvalues > 0)
