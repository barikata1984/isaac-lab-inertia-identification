"""Tests for two-stage cylinder payload model."""

import numpy as np
import pytest

from models.payloads.two_stage_cylinder import TwoStageCylinderPayload


def test_two_stage_defaults():
    """Test default two-stage cylinder payload parameters."""
    payload = TwoStageCylinderPayload()
    assert payload.lower_radius == 0.10
    assert payload.lower_height == 0.15
    assert payload.upper_radius == 0.20
    assert payload.upper_height == 0.20
    assert payload.density == 30.0


def test_two_stage_volumes():
    """Test volume calculations."""
    payload = TwoStageCylinderPayload()
    expected_lower = np.pi * 0.10**2 * 0.15
    expected_upper = np.pi * 0.20**2 * 0.20
    assert np.isclose(payload.lower_volume, expected_lower)
    assert np.isclose(payload.upper_volume, expected_upper)


def test_two_stage_masses():
    """Test mass calculations."""
    payload = TwoStageCylinderPayload(density=1000.0)
    expected_lower = 1000.0 * np.pi * 0.10**2 * 0.15
    expected_upper = 1000.0 * np.pi * 0.20**2 * 0.20
    assert np.isclose(payload.lower_mass, expected_lower)
    assert np.isclose(payload.upper_mass, expected_upper)
    assert np.isclose(payload.mass, expected_lower + expected_upper)


def test_two_stage_com_offset():
    """Test combined COM offset calculation."""
    payload = TwoStageCylinderPayload()
    # Lower cylinder COM: z = 0.15/2 = 0.075
    # Upper cylinder COM: z = 0.15 + 0.20/2 = 0.25
    # Combined COM (weighted average)
    z_lower = 0.15 / 2
    z_upper = 0.15 + 0.20 / 2
    m_lower = payload.lower_mass
    m_upper = payload.upper_mass
    m_total = m_lower + m_upper
    z_com_expected = (m_lower * z_lower + m_upper * z_upper) / m_total

    assert payload.com_offset[0] == 0.0
    assert payload.com_offset[1] == 0.0
    assert np.isclose(payload.com_offset[2], z_com_expected)


def test_two_stage_com_between_cylinders():
    """Test that COM is between lower and upper cylinder COMs."""
    payload = TwoStageCylinderPayload()
    z_com = payload.com_offset[2]
    z_lower_com = payload.lower_height / 2
    z_upper_com = payload.lower_height + payload.upper_height / 2

    # COM should be between the two cylinder COMs
    assert z_lower_com < z_com < z_upper_com


def test_two_stage_inertia_tensor_shape():
    """Test inertia tensor shape and symmetry."""
    payload = TwoStageCylinderPayload()
    I_com = payload.inertia_tensor
    I_tool0 = payload.inertia_at_tool0

    assert I_com.shape == (3, 3)
    assert I_tool0.shape == (3, 3)
    # Check symmetry
    assert np.allclose(I_com, I_com.T)
    assert np.allclose(I_tool0, I_tool0.T)


def test_two_stage_inertia_positive_definite():
    """Test that inertia tensors are positive definite."""
    payload = TwoStageCylinderPayload()
    I_com = payload.inertia_tensor
    I_tool0 = payload.inertia_at_tool0

    # All eigenvalues should be positive
    eig_com = np.linalg.eigvalsh(I_com)
    eig_tool0 = np.linalg.eigvalsh(I_tool0)
    assert np.all(eig_com > 0)
    assert np.all(eig_tool0 > 0)


def test_two_stage_phi_true():
    """Test 10-parameter inertial vector."""
    payload = TwoStageCylinderPayload()
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


def test_two_stage_mass_positive():
    """Test that all masses are positive."""
    payload = TwoStageCylinderPayload()
    assert payload.lower_mass > 0
    assert payload.upper_mass > 0
    assert payload.mass > 0


def test_two_stage_custom_dimensions():
    """Test with custom dimensions."""
    payload = TwoStageCylinderPayload(
        lower_radius=0.05,
        lower_height=0.10,
        upper_radius=0.15,
        upper_height=0.15,
        density=500.0,
    )

    assert payload.lower_radius == 0.05
    assert payload.lower_height == 0.10
    assert payload.upper_radius == 0.15
    assert payload.upper_height == 0.15
    assert payload.density == 500.0
    assert payload.mass > 0
    assert payload.phi_true.shape == (10,)


def test_two_stage_upper_larger_than_lower():
    """Test that upper cylinder is larger in default config."""
    payload = TwoStageCylinderPayload()
    # Upper should have larger radius
    assert payload.upper_radius > payload.lower_radius
    # Upper should also be heavier (larger volume despite similar height)
    assert payload.upper_mass > payload.lower_mass
