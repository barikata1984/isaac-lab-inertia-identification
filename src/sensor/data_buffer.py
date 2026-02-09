"""Time-series data buffer for inertial parameter estimation.

This module provides a buffer for collecting and managing sensor data
over time, preparing it for batch or online estimation.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .data_types import EstimationData, SensorData


class DataBuffer:
    """Buffer for collecting time-series sensor data.

    Provides functionality for:
    - Adding sensor samples over time
    - Converting to estimation format (regressor matrices)
    - Saving/loading data to/from files
    - Filtering and preprocessing

    Example:
        >>> buffer = DataBuffer(max_samples=1000)
        >>> for t in range(100):
        ...     data = SensorData(...)
        ...     buffer.add_sample(data)
        >>> A, y = buffer.get_stacked_data(kinematics)
        >>> phi = estimate(A, y)
    """

    def __init__(self, max_samples: int = 10000):
        """Initialize data buffer.

        Args:
            max_samples: Maximum number of samples to store.
                When exceeded, oldest samples are discarded.
        """
        self.max_samples = max_samples
        self._samples: List[SensorData] = []

    def __len__(self) -> int:
        """Number of samples in buffer."""
        return len(self._samples)

    def __iter__(self) -> Iterator[SensorData]:
        """Iterate over samples."""
        return iter(self._samples)

    def __getitem__(self, idx: int) -> SensorData:
        """Get sample by index."""
        return self._samples[idx]

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._samples) == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return len(self._samples) >= self.max_samples

    @property
    def timestamps(self) -> np.ndarray:
        """Get all timestamps as array."""
        if self.is_empty:
            return np.array([])
        return np.array([s.timestamp for s in self._samples])

    @property
    def duration(self) -> float:
        """Total duration of collected data in seconds."""
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1].timestamp - self._samples[0].timestamp

    def add_sample(self, data: SensorData) -> None:
        """Add a sensor data sample to the buffer.

        Args:
            data: Sensor data sample to add.
        """
        if self.is_full:
            self._samples.pop(0)
        self._samples.append(data)

    def add_samples(self, samples: List[SensorData]) -> None:
        """Add multiple samples at once.

        Args:
            samples: List of sensor data samples.
        """
        for sample in samples:
            self.add_sample(sample)

    def clear(self) -> None:
        """Clear all samples from buffer."""
        self._samples.clear()

    def get_joint_positions(self) -> np.ndarray:
        """Get all joint positions as (N, 6) array."""
        if self.is_empty:
            return np.empty((0, 6))
        return np.array([s.q for s in self._samples])

    def get_joint_velocities(self) -> np.ndarray:
        """Get all joint velocities as (N, 6) array."""
        if self.is_empty:
            return np.empty((0, 6))
        return np.array([s.dq for s in self._samples])

    def get_joint_accelerations(self) -> np.ndarray:
        """Get all joint accelerations as (N, 6) array."""
        if self.is_empty:
            return np.empty((0, 6))
        return np.array([s.ddq for s in self._samples])

    def get_wrenches(self) -> np.ndarray:
        """Get all force-torque measurements as (N, 6) array."""
        if self.is_empty:
            return np.empty((0, 6))
        return np.array([s.wrench for s in self._samples])

    def get_estimation_data(
        self,
        kinematics,
        gravity: Optional[np.ndarray] = None,
    ) -> List[EstimationData]:
        """Convert buffer to estimation data format.

        Computes regressor matrices for all samples using the provided
        kinematics instance.

        Args:
            kinematics: PinocchioKinematics instance for computing regressors.
            gravity: Gravity vector in base frame. Default [0, 0, -9.81].

        Returns:
            List of EstimationData objects.
        """
        result = []
        for sample in self._samples:
            g = gravity if gravity is not None else sample.gravity
            A = kinematics.compute_regressor(sample.q, sample.dq, sample.ddq, g)
            y = sample.wrench
            result.append(EstimationData(A=A, y=y, timestamp=sample.timestamp))
        return result

    def get_stacked_data(
        self,
        kinematics,
        gravity: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get stacked regressor matrix and observation vector.

        Useful for batch estimation methods.

        Args:
            kinematics: PinocchioKinematics instance.
            gravity: Gravity vector in base frame.

        Returns:
            Tuple of (A_stacked, y_stacked):
                - A_stacked: (N*6, 10) stacked regressor matrices
                - y_stacked: (N*6,) stacked observation vectors
        """
        if self.is_empty:
            return np.empty((0, 10)), np.empty(0)

        estimation_data = self.get_estimation_data(kinematics, gravity)
        A_list = [ed.A for ed in estimation_data]
        y_list = [ed.y for ed in estimation_data]

        A_stacked = np.vstack(A_list)
        y_stacked = np.concatenate(y_list)

        return A_stacked, y_stacked

    def compute_accelerations_from_velocities(
        self,
        filter_window: int = 5,
    ) -> None:
        """Compute joint accelerations from velocities using finite differences.

        Updates ddq values in-place using filtered numerical differentiation.

        Args:
            filter_window: Window size for Savitzky-Golay-like smoothing.
                Must be odd and >= 3.
        """
        if len(self._samples) < 3:
            return

        timestamps = self.timestamps
        velocities = self.get_joint_velocities()

        # Simple central differences with boundary handling
        n = len(self._samples)
        accelerations = np.zeros_like(velocities)

        for i in range(n):
            if i == 0:
                # Forward difference
                dt = timestamps[1] - timestamps[0]
                if dt > 0:
                    accelerations[i] = (velocities[1] - velocities[0]) / dt
            elif i == n - 1:
                # Backward difference
                dt = timestamps[-1] - timestamps[-2]
                if dt > 0:
                    accelerations[i] = (velocities[-1] - velocities[-2]) / dt
            else:
                # Central difference
                dt = timestamps[i + 1] - timestamps[i - 1]
                if dt > 0:
                    accelerations[i] = (velocities[i + 1] - velocities[i - 1]) / dt

        # Apply simple moving average filter if requested
        if filter_window >= 3:
            from scipy.ndimage import uniform_filter1d
            for j in range(6):
                accelerations[:, j] = uniform_filter1d(
                    accelerations[:, j], size=filter_window, mode='nearest'
                )

        # Update samples
        for i, sample in enumerate(self._samples):
            sample.ddq = accelerations[i]

    def subsample(self, factor: int) -> "DataBuffer":
        """Create a subsampled copy of the buffer.

        Args:
            factor: Subsampling factor. Keep every factor-th sample.

        Returns:
            New DataBuffer with subsampled data.
        """
        new_buffer = DataBuffer(max_samples=self.max_samples // factor)
        for i, sample in enumerate(self._samples):
            if i % factor == 0:
                new_buffer.add_sample(sample)
        return new_buffer

    def get_time_range(
        self,
        start_time: float,
        end_time: float,
    ) -> "DataBuffer":
        """Extract samples within a time range.

        Args:
            start_time: Start time in seconds.
            end_time: End time in seconds.

        Returns:
            New DataBuffer with samples in the specified range.
        """
        new_buffer = DataBuffer(max_samples=self.max_samples)
        for sample in self._samples:
            if start_time <= sample.timestamp <= end_time:
                new_buffer.add_sample(sample)
        return new_buffer

    def save_to_file(self, path: str) -> None:
        """Save buffer data to JSON file.

        Args:
            path: File path to save to.
        """
        data = {
            "max_samples": self.max_samples,
            "samples": [
                {
                    "timestamp": s.timestamp,
                    "q": s.q.tolist(),
                    "dq": s.dq.tolist(),
                    "ddq": s.ddq.tolist(),
                    "force": s.force.tolist(),
                    "torque": s.torque.tolist(),
                    "gravity": s.gravity.tolist(),
                }
                for s in self._samples
            ],
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, path: str) -> "DataBuffer":
        """Load buffer data from JSON file.

        Args:
            path: File path to load from.

        Returns:
            DataBuffer with loaded data.
        """
        with open(path, "r") as f:
            data = json.load(f)

        buffer = cls(max_samples=data.get("max_samples", 10000))
        for s in data["samples"]:
            sample = SensorData(
                timestamp=s["timestamp"],
                q=np.array(s["q"]),
                dq=np.array(s["dq"]),
                ddq=np.array(s["ddq"]),
                force=np.array(s["force"]),
                torque=np.array(s["torque"]),
                gravity=np.array(s.get("gravity", [0.0, 0.0, -9.81])),
            )
            buffer.add_sample(sample)

        return buffer

    def save_to_npz(self, path: str) -> None:
        """Save buffer data to compressed NumPy format.

        More efficient for large datasets than JSON.

        Args:
            path: File path to save to (.npz).
        """
        if self.is_empty:
            raise ValueError("Cannot save empty buffer")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            timestamps=self.timestamps,
            q=self.get_joint_positions(),
            dq=self.get_joint_velocities(),
            ddq=self.get_joint_accelerations(),
            wrenches=self.get_wrenches(),
            gravity=np.array([s.gravity for s in self._samples]),
            max_samples=self.max_samples,
        )

    @classmethod
    def load_from_npz(cls, path: str) -> "DataBuffer":
        """Load buffer data from NumPy format.

        Args:
            path: File path to load from (.npz).

        Returns:
            DataBuffer with loaded data.
        """
        data = np.load(path)

        buffer = cls(max_samples=int(data.get("max_samples", 10000)))

        timestamps = data["timestamps"]
        q_all = data["q"]
        dq_all = data["dq"]
        ddq_all = data["ddq"]
        wrenches = data["wrenches"]
        gravity_all = data.get("gravity", np.tile([0.0, 0.0, -9.81], (len(timestamps), 1)))

        for i in range(len(timestamps)):
            sample = SensorData(
                timestamp=float(timestamps[i]),
                q=q_all[i],
                dq=dq_all[i],
                ddq=ddq_all[i],
                force=wrenches[i, :3],
                torque=wrenches[i, 3:],
                gravity=gravity_all[i],
            )
            buffer.add_sample(sample)

        return buffer

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"DataBuffer(samples={len(self)}, "
            f"duration={self.duration:.3f}s, "
            f"max_samples={self.max_samples})"
        )
