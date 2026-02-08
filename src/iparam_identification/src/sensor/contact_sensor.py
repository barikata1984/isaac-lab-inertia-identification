"""Isaac Sim sensor interface for inertial parameter estimation.

This module provides interfaces for acquiring force/torque and state data
from Isaac Sim simulation.

For inertial parameter estimation, we need:
1. Joint states (q, dq, ddq)
2. Force/torque acting on the grasped object

In Isaac Sim, forces can be obtained via:
- Contact sensors (for contact forces)
- Inverse dynamics computation (for expected forces)
- External ROS 2 topics (for real hardware)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from .data_types import SensorData, WrenchStamped


class WrenchSourceBase(ABC):
    """Abstract base class for wrench (force/torque) data sources."""

    @abstractmethod
    def get_wrench(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current force and torque.

        Returns:
            Tuple of (force, torque):
                - force: (3,) force vector [N]
                - torque: (3,) torque vector [N·m]
        """
        pass

    @abstractmethod
    def get_wrench_stamped(self) -> WrenchStamped:
        """Get timestamped wrench."""
        pass


class IsaacSimStateCollector:
    """Collects robot state and wrench data from Isaac Sim.

    This class interfaces with Isaac Sim to collect:
    - Joint positions and velocities from the articulation
    - Forces/torques (via various methods)

    Example:
        >>> collector = IsaacSimStateCollector(world, robot_prim_path="/World/UR")
        >>> collector.setup()
        >>> data = collector.get_sensor_data()
    """

    def __init__(
        self,
        world,
        robot_prim_path: str,
        tool0_link: str = "tool0",
        physics_dt: float = 1.0 / 1000.0,
    ):
        """Initialize state collector.

        Args:
            world: Isaac Sim World instance.
            robot_prim_path: USD prim path of the robot articulation.
            tool0_link: Name of the tool0 link for velocity tracking.
            physics_dt: Physics timestep for acceleration computation.
        """
        self._world = world
        self._robot_prim_path = robot_prim_path
        self._tool0_link = tool0_link
        self._physics_dt = physics_dt

        # Will be initialized in setup()
        self._robot = None
        self._tool0_body = None

        # State history for acceleration computation
        self._prev_dq: Optional[np.ndarray] = None
        self._prev_time: Optional[float] = None

        # External wrench source (optional)
        self._wrench_source: Optional[WrenchSourceBase] = None

        # Cached values
        self._gravity = np.array([0.0, 0.0, -9.81])

    def setup(self) -> None:
        """Setup Isaac Sim interfaces.

        Must be called after world.reset().
        """
        try:
            from isaacsim.core.prims import RigidPrim
            from isaacsim.core.api.robots import Robot

            # Get robot articulation
            self._robot = self._world.scene.get_object(
                self._robot_prim_path.split("/")[-1]
            )
            if self._robot is None:
                # Try direct prim access
                from isaacsim.core.api.articulation import Articulation
                self._robot = Articulation(prim_path=self._robot_prim_path)
                self._robot.initialize()

            # Setup tool0 tracking
            tool0_path = f"{self._robot_prim_path}/{self._tool0_link}"
            try:
                self._tool0_body = RigidPrim(
                    prim_paths_expr=tool0_path,
                    name="tool0_tracking"
                )
                self._tool0_body.initialize()
            except Exception:
                # Fallback to wrist_3_link
                wrist3_path = f"{self._robot_prim_path}/wrist_3_link"
                self._tool0_body = RigidPrim(
                    prim_paths_expr=wrist3_path,
                    name="tool0_tracking"
                )
                self._tool0_body.initialize()

        except ImportError as e:
            raise RuntimeError(
                f"Failed to import Isaac Sim modules. "
                f"Make sure Isaac Sim is running: {e}"
            )

    def set_wrench_source(self, source: WrenchSourceBase) -> None:
        """Set external wrench data source.

        Args:
            source: Wrench source instance.
        """
        self._wrench_source = source

    def set_gravity(self, gravity: np.ndarray) -> None:
        """Set gravity vector.

        Args:
            gravity: Gravity vector in base frame (3,).
        """
        self._gravity = np.asarray(gravity).ravel()

    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current joint positions and velocities.

        Returns:
            Tuple of (q, dq):
                - q: (6,) joint positions [rad]
                - dq: (6,) joint velocities [rad/s]
        """
        if self._robot is None:
            raise RuntimeError("Collector not initialized. Call setup() first.")

        q = self._robot.get_joint_positions()
        dq = self._robot.get_joint_velocities()

        return np.asarray(q), np.asarray(dq)

    def get_joint_accelerations(self, dq: np.ndarray) -> np.ndarray:
        """Compute joint accelerations from velocity differences.

        Uses finite difference with the previous velocity measurement.

        Args:
            dq: Current joint velocities.

        Returns:
            (6,) joint accelerations [rad/s²]
        """
        current_time = self._world.current_time

        if self._prev_dq is None or self._prev_time is None:
            self._prev_dq = dq.copy()
            self._prev_time = current_time
            return np.zeros(6)

        dt = current_time - self._prev_time
        if dt <= 0:
            dt = self._physics_dt

        ddq = (dq - self._prev_dq) / dt

        self._prev_dq = dq.copy()
        self._prev_time = current_time

        return ddq

    def get_tool0_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get tool0 linear and angular velocity from simulation.

        Returns:
            Tuple of (linear_vel, angular_vel):
                - linear_vel: (3,) in world frame [m/s]
                - angular_vel: (3,) in world frame [rad/s]
        """
        if self._tool0_body is None:
            return np.zeros(3), np.zeros(3)

        linear_vels = self._tool0_body.get_linear_velocities()
        angular_vels = self._tool0_body.get_angular_velocities()

        if linear_vels is not None and len(linear_vels) > 0:
            return np.asarray(linear_vels[0]), np.asarray(angular_vels[0])

        return np.zeros(3), np.zeros(3)

    def get_wrench(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current force and torque.

        If an external wrench source is set, uses that.
        Otherwise, returns zeros (to be filled by inverse dynamics or other means).

        Returns:
            Tuple of (force, torque):
                - force: (3,) [N]
                - torque: (3,) [N·m]
        """
        if self._wrench_source is not None:
            return self._wrench_source.get_wrench()

        # Default: zeros (caller should compute via inverse dynamics)
        return np.zeros(3), np.zeros(3)

    def get_sensor_data(self) -> SensorData:
        """Collect complete sensor data for one timestep.

        Returns:
            SensorData instance with all measurements.
        """
        q, dq = self.get_joint_states()
        ddq = self.get_joint_accelerations(dq)
        force, torque = self.get_wrench()

        return SensorData(
            timestamp=self._world.current_time if self._world else 0.0,
            q=q,
            dq=dq,
            ddq=ddq,
            force=force,
            torque=torque,
            gravity=self._gravity.copy(),
        )

    def reset_acceleration_history(self) -> None:
        """Reset acceleration computation history.

        Call this when starting a new data collection sequence.
        """
        self._prev_dq = None
        self._prev_time = None


class InverseDynamicsWrenchSource(WrenchSourceBase):
    """Compute wrench from inverse dynamics.

    Uses the Newton-Euler equations to compute the force/torque
    that would be measured at the tool0 frame given the robot's
    motion and a known payload.

    This is useful for:
    1. Simulating force sensor measurements
    2. Validating estimation results
    """

    def __init__(self, kinematics, gravity: np.ndarray = None):
        """Initialize inverse dynamics wrench source.

        Args:
            kinematics: PinocchioKinematics instance.
            gravity: Gravity vector in base frame.
        """
        self._kinematics = kinematics
        self._gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])

        # Known payload parameters (for simulation/validation)
        self._payload_phi: Optional[np.ndarray] = None

        # Current state (must be updated externally)
        self._q: np.ndarray = np.zeros(6)
        self._dq: np.ndarray = np.zeros(6)
        self._ddq: np.ndarray = np.zeros(6)
        self._timestamp: float = 0.0

    def set_payload(self, phi: np.ndarray) -> None:
        """Set known payload parameters for wrench computation.

        Args:
            phi: (10,) inertial parameter vector.
        """
        self._payload_phi = np.asarray(phi).ravel()

    def update_state(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        timestamp: float = 0.0,
    ) -> None:
        """Update current robot state.

        Args:
            q: Joint positions (6,).
            dq: Joint velocities (6,).
            ddq: Joint accelerations (6,).
            timestamp: Current time.
        """
        self._q = np.asarray(q).ravel()
        self._dq = np.asarray(dq).ravel()
        self._ddq = np.asarray(ddq).ravel()
        self._timestamp = timestamp

    def get_wrench(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute wrench from inverse dynamics.

        Returns:
            Tuple of (force, torque) in tool0 frame.
        """
        if self._payload_phi is None:
            return np.zeros(3), np.zeros(3)

        # Compute regressor matrix
        A = self._kinematics.compute_regressor(
            self._q, self._dq, self._ddq, self._gravity
        )

        # Compute wrench: [f; τ] = A @ φ
        wrench = A @ self._payload_phi

        return wrench[:3], wrench[3:]

    def get_wrench_stamped(self) -> WrenchStamped:
        """Get timestamped wrench."""
        force, torque = self.get_wrench()
        return WrenchStamped(
            timestamp=self._timestamp,
            force=force,
            torque=torque,
            frame_id="tool0",
        )


class SimulatedForceSensor:
    """Simulated force/torque sensor for testing and validation.

    Generates synthetic wrench measurements with configurable noise
    based on known payload parameters and robot motion.
    """

    def __init__(
        self,
        kinematics,
        payload_phi: np.ndarray,
        noise_force_std: float = 0.1,
        noise_torque_std: float = 0.01,
        gravity: np.ndarray = None,
    ):
        """Initialize simulated sensor.

        Args:
            kinematics: PinocchioKinematics instance.
            payload_phi: True payload inertial parameters (10,).
            noise_force_std: Standard deviation of force noise [N].
            noise_torque_std: Standard deviation of torque noise [N·m].
            gravity: Gravity vector in base frame.
        """
        self._inv_dyn = InverseDynamicsWrenchSource(kinematics, gravity)
        self._inv_dyn.set_payload(payload_phi)

        self._noise_force_std = noise_force_std
        self._noise_torque_std = noise_torque_std

        self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def measure(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        timestamp: float = 0.0,
    ) -> WrenchStamped:
        """Generate a simulated measurement.

        Args:
            q: Joint positions (6,).
            dq: Joint velocities (6,).
            ddq: Joint accelerations (6,).
            timestamp: Current time.

        Returns:
            WrenchStamped with noisy measurement.
        """
        self._inv_dyn.update_state(q, dq, ddq, timestamp)
        force_true, torque_true = self._inv_dyn.get_wrench()

        # Add noise
        force_noise = self._rng.normal(0, self._noise_force_std, 3)
        torque_noise = self._rng.normal(0, self._noise_torque_std, 3)

        return WrenchStamped(
            timestamp=timestamp,
            force=force_true + force_noise,
            torque=torque_true + torque_noise,
            frame_id="tool0",
        )


def compute_offset_compensation_matrix(g_init: np.ndarray) -> np.ndarray:
    """Compute sensor offset compensation matrix.

    Based on Kubus et al. 2008, Eq. (7-8).

    When zeroing the sensor at initial orientation, gravitational forces
    are eliminated. This function computes the matrix that accounts for
    those eliminated forces.

    Args:
        g_init: Initial gravity vector in sensor frame (3,) when sensor
            was zeroed.

    Returns:
        (6, 10) offset compensation matrix A_ginit.
    """
    g_init = np.asarray(g_init).ravel()
    gx, gy, gz = g_init

    # A_tilde_ginit (6x4) - affects only first 4 parameters
    A_tilde = np.array([
        [gx, 0, 0, 0],
        [gy, 0, 0, 0],
        [gz, 0, 0, 0],
        [0, 0, gz, -gy],
        [0, -gz, 0, gx],
        [0, gy, -gx, 0],
    ])

    # Full matrix (6x10) with zeros for inertia parameters
    A_ginit = np.zeros((6, 10))
    A_ginit[:, :4] = A_tilde

    return A_ginit


def apply_offset_compensation(
    A: np.ndarray,
    g_init: np.ndarray,
) -> np.ndarray:
    """Apply offset compensation to regressor matrix.

    Computes A_offs = A + A_ginit as per Kubus et al. 2008, Eq. (9).

    Args:
        A: Original regressor matrix (6, 10) or stacked (N*6, 10).
        g_init: Initial gravity vector in sensor frame (3,).

    Returns:
        Offset-compensated regressor matrix.
    """
    A_ginit = compute_offset_compensation_matrix(g_init)

    if A.shape == (6, 10):
        return A + A_ginit
    else:
        # Stacked matrix: add A_ginit to each 6-row block
        n_samples = A.shape[0] // 6
        A_offs = A.copy()
        for i in range(n_samples):
            A_offs[i * 6:(i + 1) * 6] += A_ginit
        return A_offs
