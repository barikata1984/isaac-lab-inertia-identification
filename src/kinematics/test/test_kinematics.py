"""Tests for Pinocchio-based kinematics module."""

import numpy as np
import pytest

from kinematics import PinocchioKinematics, Tool0State, compute_regressor_matrix


class TestModelLoading:
    """Tests for model loading."""

    def test_load_ur5e(self):
        """Test UR5e model can be loaded."""
        kin = PinocchioKinematics.for_ur5e()
        assert kin.n_joints == 6

    def test_invalid_frame_raises(self):
        """Test that invalid frame name raises ValueError."""
        kin = PinocchioKinematics.for_ur5e()
        with pytest.raises(ValueError, match="not found"):
            PinocchioKinematics(kin.model, kin.data, "invalid_frame")


class TestForwardKinematics:
    """Tests for forward kinematics."""

    def test_fk_zero_config(self, ur5e_kin, zero_config):
        """Test FK at zero configuration."""
        pos, rot = ur5e_kin.forward_kinematics(zero_config)

        # Position should be valid
        assert pos.shape == (3,)
        assert np.isfinite(pos).all()

        # Rotation should be orthogonal
        assert rot.shape == (3, 3)
        np.testing.assert_array_almost_equal(rot @ rot.T, np.eye(3), decimal=10)

    def test_fk_random_config(self, ur5e_kin, random_config):
        """Test FK at random configuration."""
        pos, rot = ur5e_kin.forward_kinematics(random_config)

        assert pos.shape == (3,)
        assert rot.shape == (3, 3)
        np.testing.assert_array_almost_equal(rot @ rot.T, np.eye(3), decimal=10)

    def test_fk_deterministic(self, ur5e_kin, random_config):
        """Test FK gives consistent results."""
        pos1, rot1 = ur5e_kin.forward_kinematics(random_config)
        pos2, rot2 = ur5e_kin.forward_kinematics(random_config)

        np.testing.assert_array_equal(pos1, pos2)
        np.testing.assert_array_equal(rot1, rot2)


class TestVelocity:
    """Tests for velocity computation."""

    def test_zero_velocity(self, ur5e_kin, random_config):
        """Zero joint velocity should give zero tool0 velocity."""
        dq = np.zeros(6)

        lin_vel, ang_vel = ur5e_kin.tool0_velocity(random_config, dq)

        np.testing.assert_array_almost_equal(lin_vel, np.zeros(3), decimal=10)
        np.testing.assert_array_almost_equal(ang_vel, np.zeros(3), decimal=10)

    def test_velocity_shape(self, ur5e_kin, zero_config, random_velocity):
        """Velocity should return 3-element vectors."""
        lin_vel, ang_vel = ur5e_kin.tool0_velocity(zero_config, random_velocity)

        assert lin_vel.shape == (3,)
        assert ang_vel.shape == (3,)

    def test_velocity_linearity(self, ur5e_kin, random_config):
        """Velocity should be linear in dq (V = J @ dq)."""
        dq1 = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        dq2 = np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0])

        lin1, ang1 = ur5e_kin.tool0_velocity(random_config, dq1)
        lin2, ang2 = ur5e_kin.tool0_velocity(random_config, dq2)
        lin12, ang12 = ur5e_kin.tool0_velocity(random_config, dq1 + dq2)

        np.testing.assert_array_almost_equal(lin12, lin1 + lin2, decimal=10)
        np.testing.assert_array_almost_equal(ang12, ang1 + ang2, decimal=10)

    def test_velocity_scaling(self, ur5e_kin, random_config, random_velocity):
        """Velocity should scale linearly with joint velocity."""
        scale = 2.5

        lin1, ang1 = ur5e_kin.tool0_velocity(random_config, random_velocity)
        lin2, ang2 = ur5e_kin.tool0_velocity(random_config, scale * random_velocity)

        np.testing.assert_array_almost_equal(lin2, scale * lin1, decimal=10)
        np.testing.assert_array_almost_equal(ang2, scale * ang1, decimal=10)

    def test_velocity_local_vs_world_norms(
        self, ur5e_kin, random_config, random_velocity
    ):
        """LOCAL and WORLD angular velocities should have same norm."""
        lin_local, ang_local = ur5e_kin.tool0_velocity(
            random_config, random_velocity, frame="local"
        )
        lin_world, ang_world = ur5e_kin.tool0_velocity(
            random_config, random_velocity, frame="world"
        )

        # Angular velocity norms should be equal (rotation preserves norm)
        np.testing.assert_almost_equal(
            np.linalg.norm(ang_local), np.linalg.norm(ang_world), decimal=10
        )


class TestAcceleration:
    """Tests for classical acceleration computation."""

    def test_zero_velocity_zero_acceleration(self, ur5e_kin, random_config):
        """With zero velocity and zero joint acceleration, tool0 acceleration is zero."""
        dq = np.zeros(6)
        ddq = np.zeros(6)

        lin_acc, ang_acc = ur5e_kin.tool0_acceleration(random_config, dq, ddq)

        np.testing.assert_array_almost_equal(lin_acc, np.zeros(3), decimal=10)
        np.testing.assert_array_almost_equal(ang_acc, np.zeros(3), decimal=10)

    def test_acceleration_shape(
        self, ur5e_kin, zero_config, random_velocity, random_acceleration
    ):
        """Acceleration should return 3-element vectors."""
        lin_acc, ang_acc = ur5e_kin.tool0_acceleration(
            zero_config, random_velocity, random_acceleration
        )

        assert lin_acc.shape == (3,)
        assert ang_acc.shape == (3,)

    def test_acceleration_with_zero_velocity(
        self, ur5e_kin, random_config, random_acceleration
    ):
        """With zero velocity, acceleration should be J @ ddq."""
        dq = np.zeros(6)

        lin_acc, ang_acc = ur5e_kin.tool0_acceleration(
            random_config, dq, random_acceleration
        )

        # Should be non-zero when ddq is non-zero
        total_acc = np.linalg.norm(lin_acc) + np.linalg.norm(ang_acc)
        assert total_acc > 0


class TestFullState:
    """Tests for complete state computation."""

    def test_full_state_types(
        self, ur5e_kin, random_config, random_velocity, random_acceleration
    ):
        """Test compute_full_state returns valid Tool0State."""
        state = ur5e_kin.compute_full_state(
            random_config, random_velocity, random_acceleration
        )

        assert isinstance(state, Tool0State)
        assert state.position.shape == (3,)
        assert state.rotation.shape == (3, 3)
        assert state.linear_velocity.shape == (3,)
        assert state.angular_velocity.shape == (3,)
        assert state.linear_acceleration.shape == (3,)
        assert state.angular_acceleration.shape == (3,)

    def test_full_state_consistency(
        self, ur5e_kin, random_config, random_velocity, random_acceleration
    ):
        """Test that full_state matches individual method calls."""
        state = ur5e_kin.compute_full_state(
            random_config, random_velocity, random_acceleration
        )

        pos, rot = ur5e_kin.forward_kinematics(random_config)
        lin_vel, ang_vel = ur5e_kin.tool0_velocity(
            random_config, random_velocity, frame="local"
        )
        lin_acc, ang_acc = ur5e_kin.tool0_acceleration(
            random_config, random_velocity, random_acceleration, frame="local"
        )

        np.testing.assert_array_almost_equal(state.position, pos)
        np.testing.assert_array_almost_equal(state.rotation, rot)
        np.testing.assert_array_almost_equal(state.linear_velocity, lin_vel)
        np.testing.assert_array_almost_equal(state.angular_velocity, ang_vel)
        np.testing.assert_array_almost_equal(state.linear_acceleration, lin_acc)
        np.testing.assert_array_almost_equal(state.angular_acceleration, ang_acc)


class TestPhysicalPlausibility:
    """Tests for physical plausibility of results."""

    def test_position_in_workspace(self, ur5e_kin):
        """Test that FK positions are within UR5e workspace."""
        np.random.seed(123)
        for _ in range(10):
            q = np.random.uniform(-np.pi, np.pi, 6)
            pos, _ = ur5e_kin.forward_kinematics(q)

            # UR5e reach is approximately 0.85m
            distance = np.linalg.norm(pos)
            assert distance < 1.5, f"Position {pos} seems outside workspace"

    def test_rotation_is_valid(self, ur5e_kin):
        """Test that rotations are valid SO(3) matrices."""
        np.random.seed(456)
        for _ in range(10):
            q = np.random.uniform(-np.pi, np.pi, 6)
            _, rot = ur5e_kin.forward_kinematics(q)

            # Check orthogonality
            np.testing.assert_array_almost_equal(
                rot @ rot.T, np.eye(3), decimal=10
            )

            # Check determinant is +1
            np.testing.assert_almost_equal(np.linalg.det(rot), 1.0, decimal=10)


class TestRegressorMatrix:
    """Tests for regressor matrix computation."""

    def test_regressor_shape(self):
        """Test regressor matrix has correct shape (6, 10)."""
        a = np.array([1.0, 2.0, 3.0])
        alpha = np.array([0.1, 0.2, 0.3])
        omega = np.array([0.5, 0.6, 0.7])
        g = np.array([0.0, 0.0, -9.81])

        A = compute_regressor_matrix(a, alpha, omega, g)

        assert A.shape == (6, 10)

    def test_regressor_zero_angular(self):
        """Test regressor with zero angular velocity and acceleration."""
        a = np.array([0.0, 0.0, 9.81])  # Compensating gravity
        alpha = np.zeros(3)
        omega = np.zeros(3)
        g = np.array([0.0, 0.0, -9.81])

        A = compute_regressor_matrix(a, alpha, omega, g)

        # With omega=0 and alpha=0:
        # - Force part should only have (a-g) in first column
        # - Inertia columns (4-9) in torque rows should be mostly zero
        # First column should be [ax-gx, ay-gy, az-gz, 0, 0, 0]
        expected_col0 = np.array([0.0, 0.0, 9.81 + 9.81, 0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(A[:, 0], expected_col0)

        # Inertia columns in force rows (0-2, cols 4-9) should be zero
        np.testing.assert_array_almost_equal(A[:3, 4:], np.zeros((3, 6)))

    def test_regressor_static_case(self):
        """Test static case: no motion, only gravity."""
        g = np.array([0.0, 0.0, -9.81])
        a = np.zeros(3)
        alpha = np.zeros(3)
        omega = np.zeros(3)

        A = compute_regressor_matrix(a, alpha, omega, g)

        # First column should be [-gx, -gy, -gz, 0, 0, 0] = [0, 0, 9.81, 0, 0, 0]
        np.testing.assert_array_almost_equal(
            A[:, 0], np.array([0.0, 0.0, 9.81, 0.0, 0.0, 0.0])
        )

    def test_regressor_force_torque_relation(self):
        """Test that A @ phi gives expected force/torque for known parameters."""
        # Simple case: unit mass at origin, no rotation
        m = 1.0
        cx, cy, cz = 0.0, 0.0, 0.0
        Ixx, Ixy, Ixz, Iyy, Iyz, Izz = 0.1, 0.0, 0.0, 0.1, 0.0, 0.1

        phi = np.array([m, m * cx, m * cy, m * cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz])

        # Static in gravity field
        a = np.array([0.0, 0.0, 0.0])
        alpha = np.zeros(3)
        omega = np.zeros(3)
        g = np.array([0.0, 0.0, -9.81])

        A = compute_regressor_matrix(a, alpha, omega, g)
        f_tau = A @ phi

        # Expected: f = m*(a-g) = m*[0,0,9.81], tau = 0 (no rotation, CoM at origin)
        expected_f = np.array([0.0, 0.0, 9.81])
        expected_tau = np.zeros(3)

        np.testing.assert_array_almost_equal(f_tau[:3], expected_f, decimal=5)
        np.testing.assert_array_almost_equal(f_tau[3:], expected_tau, decimal=5)

    def test_regressor_with_offset_com(self):
        """Test regressor with center of mass offset."""
        m = 2.0
        cx, cy, cz = 0.1, 0.0, 0.0  # CoM offset in x
        phi = np.array([m, m * cx, m * cy, m * cz, 0, 0, 0, 0, 0, 0])

        # Static in gravity field pointing down (-z)
        a = np.zeros(3)
        alpha = np.zeros(3)
        omega = np.zeros(3)
        g = np.array([0.0, 0.0, -9.81])

        A = compute_regressor_matrix(a, alpha, omega, g)
        f_tau = A @ phi

        # Force should be m*(-g) = [0, 0, m*9.81]
        expected_f = np.array([0.0, 0.0, m * 9.81])
        np.testing.assert_array_almost_equal(f_tau[:3], expected_f, decimal=5)

        # Torque about sensor origin due to gravity on offset mass
        # tau = c x f = [cx, cy, cz] x [0, 0, m*9.81]
        # = [cy*m*9.81 - cz*0, cz*0 - cx*m*9.81, cx*0 - cy*0]
        # = [0, -cx*m*9.81, 0] = [0, -0.1*2*9.81, 0]
        expected_tau_y = -cx * m * 9.81
        np.testing.assert_almost_equal(f_tau[4], expected_tau_y, decimal=5)


class TestPinocchioKinematicsRegressor:
    """Tests for PinocchioKinematics.compute_regressor method."""

    def test_compute_regressor_shape(self, ur5e_kin, random_config):
        """Test that compute_regressor returns (6, 10) matrix."""
        dq = np.zeros(6)
        ddq = np.zeros(6)

        A = ur5e_kin.compute_regressor(random_config, dq, ddq)

        assert A.shape == (6, 10)

    def test_compute_regressor_static(self, ur5e_kin, random_config):
        """Test regressor in static configuration."""
        dq = np.zeros(6)
        ddq = np.zeros(6)

        A = ur5e_kin.compute_regressor(random_config, dq, ddq)

        # Inertia columns in force rows should be zero (no rotation)
        np.testing.assert_array_almost_equal(A[:3, 4:], np.zeros((3, 6)))

    def test_compute_regressor_consistency(
        self, ur5e_kin, random_config, random_velocity, random_acceleration
    ):
        """Test that compute_regressor is consistent with individual kinematic calls."""
        gravity = np.array([0.0, 0.0, -9.81])

        # Compute regressor using the method
        A1 = ur5e_kin.compute_regressor(
            random_config, random_velocity, random_acceleration, gravity
        )

        # Compute manually
        _, rot = ur5e_kin.forward_kinematics(random_config)
        _, omega = ur5e_kin.tool0_velocity(random_config, random_velocity, "local")
        a, alpha = ur5e_kin.tool0_acceleration(
            random_config, random_velocity, random_acceleration, "local"
        )
        g_local = rot.T @ gravity

        A2 = compute_regressor_matrix(a, alpha, omega, g_local)

        np.testing.assert_array_almost_equal(A1, A2, decimal=8)

    def test_compute_regressor_deterministic(
        self, ur5e_kin, random_config, random_velocity, random_acceleration
    ):
        """Test that regressor computation is deterministic."""
        A1 = ur5e_kin.compute_regressor(
            random_config, random_velocity, random_acceleration
        )
        A2 = ur5e_kin.compute_regressor(
            random_config, random_velocity, random_acceleration
        )

        np.testing.assert_array_equal(A1, A2)
