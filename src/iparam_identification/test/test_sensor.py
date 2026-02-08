"""Unit tests for sensor module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from iparam_identification.sensor import (
    DataBuffer,
    EstimationData,
    EstimationResult,
    SensorData,
    WrenchStamped,
    apply_offset_compensation,
    compute_offset_compensation_matrix,
)


class TestSensorData:
    """Tests for SensorData class."""

    def test_creation(self, sample_joint_state, sample_wrench, gravity):
        """Test SensorData creation with valid inputs."""
        data = SensorData(
            timestamp=0.5,
            q=sample_joint_state["q"],
            dq=sample_joint_state["dq"],
            ddq=sample_joint_state["ddq"],
            force=sample_wrench["force"],
            torque=sample_wrench["torque"],
            gravity=gravity,
        )

        assert data.timestamp == 0.5
        assert data.q.shape == (6,)
        assert data.dq.shape == (6,)
        assert data.ddq.shape == (6,)
        assert data.force.shape == (3,)
        assert data.torque.shape == (3,)
        assert data.gravity.shape == (3,)

    def test_wrench_property(self, sample_joint_state, sample_wrench, gravity):
        """Test wrench property returns concatenated force-torque."""
        data = SensorData(
            timestamp=0.0,
            q=sample_joint_state["q"],
            dq=sample_joint_state["dq"],
            ddq=sample_joint_state["ddq"],
            force=sample_wrench["force"],
            torque=sample_wrench["torque"],
            gravity=gravity,
        )

        wrench = data.wrench
        assert wrench.shape == (6,)
        np.testing.assert_array_equal(wrench[:3], sample_wrench["force"])
        np.testing.assert_array_equal(wrench[3:], sample_wrench["torque"])

    def test_invalid_shape_raises(self, sample_wrench, gravity):
        """Test that invalid shapes raise ValueError."""
        with pytest.raises(ValueError):
            SensorData(
                timestamp=0.0,
                q=np.zeros(5),  # Wrong shape
                dq=np.zeros(6),
                ddq=np.zeros(6),
                force=sample_wrench["force"],
                torque=sample_wrench["torque"],
                gravity=gravity,
            )

    def test_default_gravity(self, sample_joint_state, sample_wrench):
        """Test default gravity is applied."""
        data = SensorData(
            timestamp=0.0,
            q=sample_joint_state["q"],
            dq=sample_joint_state["dq"],
            ddq=sample_joint_state["ddq"],
            force=sample_wrench["force"],
            torque=sample_wrench["torque"],
        )

        np.testing.assert_array_almost_equal(data.gravity, [0, 0, -9.81])


class TestEstimationData:
    """Tests for EstimationData class."""

    def test_creation(self):
        """Test EstimationData creation."""
        A = np.random.randn(6, 10)
        y = np.random.randn(6)

        data = EstimationData(A=A, y=y, timestamp=1.0)

        assert data.A.shape == (6, 10)
        assert data.y.shape == (6,)
        assert data.timestamp == 1.0

    def test_invalid_shape_raises(self):
        """Test invalid shapes raise ValueError."""
        with pytest.raises(ValueError):
            EstimationData(A=np.zeros((5, 10)), y=np.zeros(6))

        with pytest.raises(ValueError):
            EstimationData(A=np.zeros((6, 10)), y=np.zeros(5))


class TestEstimationResult:
    """Tests for EstimationResult class."""

    def test_creation(self, sample_phi):
        """Test EstimationResult creation."""
        result = EstimationResult(
            phi=sample_phi,
            condition_number=10.5,
            residual_norm=0.001,
            n_samples=100,
        )

        assert result.phi.shape == (10,)
        assert result.condition_number == 10.5
        assert result.residual_norm == 0.001
        assert result.n_samples == 100

    def test_mass_property(self, sample_phi):
        """Test mass extraction from phi."""
        result = EstimationResult(phi=sample_phi)
        assert result.mass == 1.0

    def test_center_of_mass_property(self, sample_phi):
        """Test center of mass extraction."""
        result = EstimationResult(phi=sample_phi)
        com = result.center_of_mass

        assert com.shape == (3,)
        np.testing.assert_array_almost_equal(com, [0, 0, 0.05])

    def test_inertia_matrix_property(self, sample_phi):
        """Test inertia matrix extraction."""
        result = EstimationResult(phi=sample_phi)
        I = result.inertia_matrix

        assert I.shape == (3, 3)
        # Check symmetry
        np.testing.assert_array_almost_equal(I, I.T)
        # Check diagonal values
        np.testing.assert_almost_equal(I[0, 0], 0.01)
        np.testing.assert_almost_equal(I[1, 1], 0.01)
        np.testing.assert_almost_equal(I[2, 2], 0.01)

    def test_inertia_at_com(self, sample_phi):
        """Test inertia transformation to CoM."""
        result = EstimationResult(phi=sample_phi)
        I_com = result.inertia_at_com

        # For CoM at (0, 0, 0.05), parallel axis theorem:
        # I_com = I_sensor - m * (c^T c * I - c c^T)
        # Since c = (0, 0, 0.05), c^T c = 0.0025
        # The z-component remains unchanged for diagonal elements
        assert I_com.shape == (3, 3)

    def test_to_dict(self, sample_phi):
        """Test serialization to dictionary."""
        result = EstimationResult(
            phi=sample_phi,
            condition_number=10.0,
            residual_norm=0.01,
            n_samples=50,
        )

        d = result.to_dict()
        assert "mass" in d
        assert "center_of_mass" in d
        assert "inertia_matrix" in d
        assert d["mass"] == 1.0
        assert d["n_samples"] == 50

    def test_string_representation(self, sample_phi):
        """Test string output."""
        result = EstimationResult(phi=sample_phi)
        s = str(result)

        assert "Mass:" in s
        assert "Center of Mass:" in s
        assert "Inertia Matrix" in s


class TestWrenchStamped:
    """Tests for WrenchStamped class."""

    def test_creation(self):
        """Test WrenchStamped creation."""
        wrench = WrenchStamped(
            timestamp=1.5,
            force=np.array([1, 2, 3]),
            torque=np.array([0.1, 0.2, 0.3]),
            frame_id="tool0",
        )

        assert wrench.timestamp == 1.5
        assert wrench.frame_id == "tool0"
        assert wrench.force.shape == (3,)
        assert wrench.torque.shape == (3,)

    def test_wrench_property(self):
        """Test combined wrench property."""
        wrench = WrenchStamped(
            timestamp=0.0,
            force=np.array([1, 2, 3]),
            torque=np.array([4, 5, 6]),
        )

        combined = wrench.wrench
        np.testing.assert_array_equal(combined, [1, 2, 3, 4, 5, 6])


class TestDataBuffer:
    """Tests for DataBuffer class."""

    def _make_sample(self, t: float) -> SensorData:
        """Create a sample SensorData for testing."""
        return SensorData(
            timestamp=t,
            q=np.zeros(6) + t,
            dq=np.zeros(6) + t * 0.1,
            ddq=np.zeros(6) + t * 0.01,
            force=np.array([0, 0, 9.81]),
            torque=np.zeros(3),
        )

    def test_add_sample(self):
        """Test adding samples to buffer."""
        buffer = DataBuffer(max_samples=100)

        assert buffer.is_empty
        assert len(buffer) == 0

        buffer.add_sample(self._make_sample(0.0))

        assert not buffer.is_empty
        assert len(buffer) == 1

    def test_max_samples_limit(self):
        """Test buffer respects max_samples limit."""
        buffer = DataBuffer(max_samples=5)

        for i in range(10):
            buffer.add_sample(self._make_sample(float(i)))

        assert len(buffer) == 5
        # Should keep most recent
        assert buffer[0].timestamp == 5.0

    def test_clear(self):
        """Test clearing buffer."""
        buffer = DataBuffer()
        buffer.add_sample(self._make_sample(0.0))
        buffer.add_sample(self._make_sample(1.0))

        buffer.clear()
        assert buffer.is_empty

    def test_timestamps_property(self):
        """Test timestamps extraction."""
        buffer = DataBuffer()
        for i in range(5):
            buffer.add_sample(self._make_sample(float(i) * 0.1))

        timestamps = buffer.timestamps
        np.testing.assert_array_almost_equal(timestamps, [0.0, 0.1, 0.2, 0.3, 0.4])

    def test_duration_property(self):
        """Test duration calculation."""
        buffer = DataBuffer()
        buffer.add_sample(self._make_sample(1.0))
        buffer.add_sample(self._make_sample(2.5))

        assert buffer.duration == 1.5

    def test_get_joint_positions(self):
        """Test extracting joint positions."""
        buffer = DataBuffer()
        for i in range(3):
            buffer.add_sample(self._make_sample(float(i)))

        positions = buffer.get_joint_positions()
        assert positions.shape == (3, 6)

    def test_get_wrenches(self):
        """Test extracting wrenches."""
        buffer = DataBuffer()
        for i in range(3):
            buffer.add_sample(self._make_sample(float(i)))

        wrenches = buffer.get_wrenches()
        assert wrenches.shape == (3, 6)

    def test_save_load_json(self):
        """Test saving and loading from JSON."""
        buffer = DataBuffer()
        for i in range(5):
            buffer.add_sample(self._make_sample(float(i) * 0.1))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_data.json"
            buffer.save_to_file(str(path))

            loaded = DataBuffer.load_from_file(str(path))

            assert len(loaded) == len(buffer)
            np.testing.assert_array_almost_equal(
                loaded.timestamps, buffer.timestamps
            )

    def test_save_load_npz(self):
        """Test saving and loading from NPZ format."""
        buffer = DataBuffer()
        for i in range(5):
            buffer.add_sample(self._make_sample(float(i) * 0.1))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_data.npz"
            buffer.save_to_npz(str(path))

            loaded = DataBuffer.load_from_npz(str(path))

            assert len(loaded) == len(buffer)
            np.testing.assert_array_almost_equal(
                loaded.get_joint_positions(),
                buffer.get_joint_positions()
            )

    def test_subsample(self):
        """Test subsampling functionality."""
        buffer = DataBuffer()
        for i in range(10):
            buffer.add_sample(self._make_sample(float(i)))

        subsampled = buffer.subsample(factor=2)

        assert len(subsampled) == 5
        np.testing.assert_array_almost_equal(
            subsampled.timestamps, [0, 2, 4, 6, 8]
        )

    def test_get_time_range(self):
        """Test time range extraction."""
        buffer = DataBuffer()
        for i in range(10):
            buffer.add_sample(self._make_sample(float(i)))

        subset = buffer.get_time_range(3.0, 7.0)

        assert len(subset) == 5
        assert subset[0].timestamp == 3.0
        assert subset[-1].timestamp == 7.0

    def test_iteration(self):
        """Test buffer iteration."""
        buffer = DataBuffer()
        for i in range(3):
            buffer.add_sample(self._make_sample(float(i)))

        timestamps = [s.timestamp for s in buffer]
        assert timestamps == [0.0, 1.0, 2.0]


class TestOffsetCompensation:
    """Tests for sensor offset compensation functions."""

    def test_compute_offset_matrix_shape(self, gravity):
        """Test offset matrix has correct shape."""
        A_ginit = compute_offset_compensation_matrix(gravity)
        assert A_ginit.shape == (6, 10)

    def test_compute_offset_matrix_zeros_for_inertia(self, gravity):
        """Test inertia columns are zero in offset matrix."""
        A_ginit = compute_offset_compensation_matrix(gravity)
        # Columns 4-9 (inertia parameters) should be zero
        np.testing.assert_array_equal(A_ginit[:, 4:], np.zeros((6, 6)))

    def test_apply_offset_single_matrix(self, gravity):
        """Test offset application to single regressor matrix."""
        A = np.random.randn(6, 10)
        A_offs = apply_offset_compensation(A, gravity)

        assert A_offs.shape == (6, 10)
        # Should be different from original
        assert not np.allclose(A, A_offs)

    def test_apply_offset_stacked_matrix(self, gravity):
        """Test offset application to stacked regressor matrices."""
        n_samples = 5
        A_stacked = np.random.randn(n_samples * 6, 10)
        A_offs = apply_offset_compensation(A_stacked, gravity)

        assert A_offs.shape == (n_samples * 6, 10)

    def test_offset_compensation_effect(self):
        """Test that offset compensation has expected structure."""
        g_init = np.array([0, 0, -9.81])
        A_ginit = compute_offset_compensation_matrix(g_init)

        # First column (mass) should have g_init values in force rows
        np.testing.assert_array_almost_equal(A_ginit[:3, 0], g_init)

        # Torque rows for first column should be zero
        np.testing.assert_array_equal(A_ginit[3:, 0], [0, 0, 0])


class TestSimulatedForceSensor:
    """Tests for SimulatedForceSensor."""

    def test_measure_returns_wrench(self, kinematics, sample_phi, sample_joint_state):
        """Test that measure returns valid WrenchStamped."""
        from iparam_identification.sensor import SimulatedForceSensor

        sensor = SimulatedForceSensor(
            kinematics=kinematics,
            payload_phi=sample_phi,
            noise_force_std=0.0,  # No noise for deterministic test
            noise_torque_std=0.0,
        )

        wrench = sensor.measure(
            q=sample_joint_state["q"],
            dq=sample_joint_state["dq"],
            ddq=sample_joint_state["ddq"],
            timestamp=1.0,
        )

        assert isinstance(wrench, WrenchStamped)
        assert wrench.force.shape == (3,)
        assert wrench.torque.shape == (3,)
        assert wrench.timestamp == 1.0

    def test_noise_affects_measurement(self, kinematics, sample_phi, sample_joint_state):
        """Test that noise is added to measurements."""
        from iparam_identification.sensor import SimulatedForceSensor

        sensor_noisy = SimulatedForceSensor(
            kinematics=kinematics,
            payload_phi=sample_phi,
            noise_force_std=1.0,
            noise_torque_std=0.1,
        )
        sensor_noisy.set_seed(42)

        sensor_clean = SimulatedForceSensor(
            kinematics=kinematics,
            payload_phi=sample_phi,
            noise_force_std=0.0,
            noise_torque_std=0.0,
        )

        wrench_noisy = sensor_noisy.measure(
            q=sample_joint_state["q"],
            dq=sample_joint_state["dq"],
            ddq=sample_joint_state["ddq"],
        )

        wrench_clean = sensor_clean.measure(
            q=sample_joint_state["q"],
            dq=sample_joint_state["dq"],
            ddq=sample_joint_state["ddq"],
        )

        # Noisy measurement should differ from clean
        assert not np.allclose(wrench_noisy.force, wrench_clean.force)


class TestInverseDynamicsWrenchSource:
    """Tests for InverseDynamicsWrenchSource."""

    def test_wrench_computation(self, kinematics, sample_phi, sample_joint_state):
        """Test wrench computation from inverse dynamics."""
        from iparam_identification.sensor import InverseDynamicsWrenchSource

        source = InverseDynamicsWrenchSource(kinematics)
        source.set_payload(sample_phi)
        source.update_state(
            q=sample_joint_state["q"],
            dq=sample_joint_state["dq"],
            ddq=sample_joint_state["ddq"],
        )

        force, torque = source.get_wrench()

        assert force.shape == (3,)
        assert torque.shape == (3,)

    def test_no_payload_returns_zeros(self, kinematics, sample_joint_state):
        """Test that no payload returns zero wrench."""
        from iparam_identification.sensor import InverseDynamicsWrenchSource

        source = InverseDynamicsWrenchSource(kinematics)
        # Don't set payload
        source.update_state(
            q=sample_joint_state["q"],
            dq=sample_joint_state["dq"],
            ddq=sample_joint_state["ddq"],
        )

        force, torque = source.get_wrench()

        np.testing.assert_array_equal(force, np.zeros(3))
        np.testing.assert_array_equal(torque, np.zeros(3))
