"""Pinocchio-based kinematics computation for UR robots.

This module provides forward kinematics, velocity, and acceleration
computation using the Pinocchio library and ur_description URDF.

Output convention: [linear, angular] (v, omega) for velocity/acceleration.
This matches Pinocchio and Isaac Sim conventions.
"""

from dataclasses import dataclass
from typing import Literal, Tuple
import subprocess

import numpy as np
import pinocchio as pin


@dataclass
class Tool0State:
    """Tool0 frame state containing pose, velocity, and acceleration.

    All velocities and accelerations are expressed in tool0 (LOCAL) frame
    unless otherwise specified.
    Uses [linear, angular] convention: [v, omega] for velocity,
    [a, alpha] for acceleration.
    """

    position: np.ndarray  # (3,) position in base frame
    rotation: np.ndarray  # (3, 3) rotation matrix (base -> tool0)
    linear_velocity: np.ndarray  # (3,) linear velocity
    angular_velocity: np.ndarray  # (3,) angular velocity
    linear_acceleration: np.ndarray  # (3,) classical linear acceleration
    angular_acceleration: np.ndarray  # (3,) angular acceleration


def load_ur_model(ur_type: str = "ur5e") -> Tuple[pin.Model, pin.Data]:
    """Load UR robot model from ur_description package.

    Args:
        ur_type: Robot type (e.g., "ur5e", "ur3", "ur10").

    Returns:
        Tuple of (model, data).

    Raises:
        RuntimeError: If xacro processing or model loading fails.
    """
    import os
    import tempfile

    # First, try to use a pre-generated URDF file
    cache_dir = os.path.join(tempfile.gettempdir(), "ur_urdf_cache")
    cached_urdf_path = os.path.join(cache_dir, f"{ur_type}.urdf")

    if os.path.exists(cached_urdf_path):
        # Use cached URDF
        try:
            model = pin.buildModelFromUrdf(cached_urdf_path)
            data = model.createData()
            return model, data
        except Exception:
            pass  # Fall through to regenerate

    from ament_index_python.packages import get_package_share_directory

    ur_description_path = get_package_share_directory("ur_description")
    xacro_file = f"{ur_description_path}/urdf/ur.urdf.xacro"

    # Generate URDF from xacro with clean environment to avoid
    # Python version conflicts when running in Isaac Sim
    env = os.environ.copy()
    # Remove Isaac Sim Python paths to use system Python for xacro
    pythonpath = env.get("PYTHONPATH", "")
    filtered_paths = [
        p for p in pythonpath.split(os.pathsep)
        if "isaac-sim" not in p and p
    ]
    env["PYTHONPATH"] = os.pathsep.join(filtered_paths)
    # Use system Python
    env.pop("LD_LIBRARY_PATH", None)

    try:
        result = subprocess.run(
            ["xacro", xacro_file, f"ur_type:={ur_type}", f"name:={ur_type}"],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to process xacro: {e.stderr}") from e

    urdf_string = result.stdout

    # Cache the URDF for future use
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cached_urdf_path, "w") as f:
            f.write(urdf_string)
    except Exception:
        pass  # Caching is optional

    # Load into Pinocchio
    model = pin.buildModelFromXML(urdf_string)
    data = model.createData()

    return model, data


class PinocchioKinematics:
    """Pinocchio-based kinematics computation class.

    Computes forward kinematics, velocity, and acceleration for UR robots
    using Pinocchio library.
    """

    def __init__(
        self,
        model: pin.Model,
        data: pin.Data,
        tool0_frame_name: str = "tool0",
    ):
        """Initialize kinematics with Pinocchio model.

        Args:
            model: Pinocchio model.
            data: Pinocchio data.
            tool0_frame_name: Name of the tool0 frame in the model.

        Raises:
            ValueError: If the specified frame is not found.
        """
        self.model = model
        self.data = data
        self.tool0_frame_id = model.getFrameId(tool0_frame_name)

        if self.tool0_frame_id >= model.nframes:
            available_frames = [
                model.frames[i].name for i in range(model.nframes)
            ]
            raise ValueError(
                f"Frame '{tool0_frame_name}' not found in model. "
                f"Available frames: {available_frames}"
            )

        self._n_joints = model.nq

    @classmethod
    def from_urdf_path(
        cls,
        urdf_path: str,
        tool0_frame_name: str = "tool0",
    ) -> "PinocchioKinematics":
        """Create kinematics instance from a URDF file path.

        Loads the Pinocchio model directly from a URDF file without
        requiring ROS2 (ament_index_python, xacro, ur_description).

        Args:
            urdf_path: Absolute path to the .urdf file.
            tool0_frame_name: Name of the end-effector frame in the URDF.

        Returns:
            PinocchioKinematics instance.

        Raises:
            FileNotFoundError: If urdf_path does not exist.
            RuntimeError: If Pinocchio fails to parse the URDF.
        """
        import os

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        try:
            model = pin.buildModelFromUrdf(urdf_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load URDF: {e}") from e

        data = model.createData()
        return cls(model, data, tool0_frame_name)

    @classmethod
    def for_ur5e(cls) -> "PinocchioKinematics":
        """Create kinematics instance for UR5e robot.

        Returns:
            PinocchioKinematics instance configured for UR5e.
        """
        model, data = load_ur_model("ur5e")
        return cls(model, data)

    @classmethod
    def for_ur_type(cls, ur_type: str) -> "PinocchioKinematics":
        """Create kinematics instance for specified UR robot type.

        Args:
            ur_type: Robot type (e.g., "ur5e", "ur3", "ur10").

        Returns:
            PinocchioKinematics instance.
        """
        model, data = load_ur_model(ur_type)
        return cls(model, data)

    @property
    def n_joints(self) -> int:
        """Number of joints in the robot model."""
        return self._n_joints

    def forward_kinematics(
        self,
        q: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics to tool0 frame.

        Args:
            q: Joint positions (n,).

        Returns:
            Tuple of (position, rotation):
                - position: (3,) tool0 position in base frame.
                - rotation: (3, 3) rotation matrix from base to tool0.
        """
        q = np.asarray(q).ravel()

        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.tool0_frame_id)

        oMf = self.data.oMf[self.tool0_frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def tool0_velocity(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        frame: Literal["local", "world"] = "local",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tool0 classical velocity.

        Args:
            q: Joint positions (n,).
            dq: Joint velocities (n,).
            frame: Reference frame ("local" = tool0, "world" = base).

        Returns:
            Tuple of (linear_velocity, angular_velocity):
                - linear_velocity: (3,) linear velocity.
                - angular_velocity: (3,) angular velocity.
        """
        q = np.asarray(q).ravel()
        dq = np.asarray(dq).ravel()

        pin.forwardKinematics(self.model, self.data, q, dq)
        pin.updateFramePlacement(self.model, self.data, self.tool0_frame_id)

        if frame == "local":
            ref_frame = pin.LOCAL
        else:
            ref_frame = pin.WORLD

        velocity = pin.getFrameVelocity(
            self.model, self.data, self.tool0_frame_id, ref_frame
        )

        return velocity.linear.copy(), velocity.angular.copy()

    def tool0_acceleration(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        frame: Literal["local", "world"] = "local",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute tool0 classical acceleration.

        Classical acceleration includes centrifugal/Coriolis effects.
        This is the actual acceleration of the frame origin (d²p/dt²),
        not the spatial acceleration.

        Args:
            q: Joint positions (n,).
            dq: Joint velocities (n,).
            ddq: Joint accelerations (n,).
            frame: Reference frame ("local" = tool0, "world" = base).

        Returns:
            Tuple of (linear_acceleration, angular_acceleration):
                - linear_acceleration: (3,) classical linear acceleration.
                - angular_acceleration: (3,) angular acceleration.
        """
        q = np.asarray(q).ravel()
        dq = np.asarray(dq).ravel()
        ddq = np.asarray(ddq).ravel()

        pin.forwardKinematics(self.model, self.data, q, dq, ddq)
        pin.updateFramePlacement(self.model, self.data, self.tool0_frame_id)

        if frame == "local":
            ref_frame = pin.LOCAL
        else:
            ref_frame = pin.WORLD

        acceleration = pin.getFrameClassicalAcceleration(
            self.model, self.data, self.tool0_frame_id, ref_frame
        )

        return acceleration.linear.copy(), acceleration.angular.copy()

    def compute_full_state(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
    ) -> Tool0State:
        """Compute complete tool0 state in LOCAL (tool0) frame.

        This is more efficient than calling individual methods when
        all quantities are needed.

        Args:
            q: Joint positions (n,).
            dq: Joint velocities (n,).
            ddq: Joint accelerations (n,).

        Returns:
            Tool0State containing pose, velocity, and acceleration.
        """
        q = np.asarray(q).ravel()
        dq = np.asarray(dq).ravel()
        ddq = np.asarray(ddq).ravel()

        # Single forward kinematics call with all derivatives
        pin.forwardKinematics(self.model, self.data, q, dq, ddq)
        pin.updateFramePlacement(self.model, self.data, self.tool0_frame_id)

        # Get pose
        oMf = self.data.oMf[self.tool0_frame_id]

        # Get velocity in LOCAL frame
        velocity = pin.getFrameVelocity(
            self.model, self.data, self.tool0_frame_id, pin.LOCAL
        )

        # Get classical acceleration in LOCAL frame
        acceleration = pin.getFrameClassicalAcceleration(
            self.model, self.data, self.tool0_frame_id, pin.LOCAL
        )

        return Tool0State(
            position=oMf.translation.copy(),
            rotation=oMf.rotation.copy(),
            linear_velocity=velocity.linear.copy(),
            angular_velocity=velocity.angular.copy(),
            linear_acceleration=acceleration.linear.copy(),
            angular_acceleration=acceleration.angular.copy(),
        )

    def compute_regressor(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        ddq: np.ndarray,
        gravity: np.ndarray = None,
    ) -> np.ndarray:
        """Compute regressor matrix for inertial parameter estimation.

        Computes the 6x10 regressor matrix A such that:
            [f; τ] = A @ φ
        where φ = [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]^T

        Based on Kubus et al. 2008, "On-line estimation of inertial parameters
        using a recursive total least-squares approach", Eq. (6).

        Args:
            q: Joint positions (n,).
            dq: Joint velocities (n,).
            ddq: Joint accelerations (n,).
            gravity: Gravity vector in base frame (3,). Default [0, 0, -9.81].

        Returns:
            (6, 10) regressor matrix in tool0 frame.
        """
        if gravity is None:
            gravity = np.array([0.0, 0.0, -9.81])
        gravity = np.asarray(gravity).ravel()

        q = np.asarray(q).ravel()
        dq = np.asarray(dq).ravel()
        ddq = np.asarray(ddq).ravel()

        # Compute kinematics
        pin.forwardKinematics(self.model, self.data, q, dq, ddq)
        pin.updateFramePlacement(self.model, self.data, self.tool0_frame_id)

        # Get rotation matrix (base -> tool0)
        oMf = self.data.oMf[self.tool0_frame_id]
        R = oMf.rotation

        # Get velocity in LOCAL frame
        velocity = pin.getFrameVelocity(
            self.model, self.data, self.tool0_frame_id, pin.LOCAL
        )
        omega = velocity.angular

        # Get classical acceleration in LOCAL frame
        acceleration = pin.getFrameClassicalAcceleration(
            self.model, self.data, self.tool0_frame_id, pin.LOCAL
        )
        a = acceleration.linear
        alpha = acceleration.angular

        # Transform gravity to tool0 frame: g_tool0 = R^T @ g_base
        g = R.T @ gravity

        return compute_regressor_matrix(a, alpha, omega, g)


def compute_regressor_matrix(
    a: np.ndarray,
    alpha: np.ndarray,
    omega: np.ndarray,
    g: np.ndarray,
) -> np.ndarray:
    """Compute regressor matrix for inertial parameter estimation.

    Computes the 6x10 regressor matrix A such that:
        [f; τ] = A @ φ
    where φ = [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]^T

    Based on Kubus et al. 2008, "On-line estimation of inertial parameters
    using a recursive total least-squares approach", Eq. (6).

    The matrix relates measured forces/torques to inertial parameters
    through the Newton-Euler equations.

    Args:
        a: Linear acceleration in sensor/tool0 frame (3,).
        alpha: Angular acceleration in sensor/tool0 frame (3,).
        omega: Angular velocity in sensor/tool0 frame (3,).
        g: Gravity vector in sensor/tool0 frame (3,).

    Returns:
        (6, 10) regressor matrix A.

    Note:
        - First 3 rows correspond to force equations
        - Last 3 rows correspond to torque equations
        - Parameter order: [m, m*cx, m*cy, m*cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    """
    a = np.asarray(a).ravel()
    alpha = np.asarray(alpha).ravel()
    omega = np.asarray(omega).ravel()
    g = np.asarray(g).ravel()

    ax, ay, az = a
    alphax, alphay, alphaz = alpha
    wx, wy, wz = omega
    gx, gy, gz = g

    # Build 6x10 regressor matrix according to Eq. (6)
    A = np.zeros((6, 10))

    # Row 1 (force x): f_x equation
    A[0, 0] = ax - gx                    # m
    A[0, 1] = -wy**2 - wz**2             # m*cx
    A[0, 2] = wx * wy - alphaz           # m*cy
    A[0, 3] = wx * wz + alphay           # m*cz

    # Row 2 (force y): f_y equation
    A[1, 0] = ay - gy                    # m
    A[1, 1] = wx * wy + alphaz           # m*cx
    A[1, 2] = -wx**2 - wz**2             # m*cy
    A[1, 3] = wy * wz - alphax           # m*cz

    # Row 3 (force z): f_z equation
    A[2, 0] = az - gz                    # m
    A[2, 1] = wx * wz - alphay           # m*cx
    A[2, 2] = wy * wz + alphax           # m*cy
    A[2, 3] = -wy**2 - wx**2             # m*cz

    # Row 4 (torque x): τ_x equation
    A[3, 1] = 0                          # m*cx
    A[3, 2] = az - gz                    # m*cy
    A[3, 3] = gy - ay                    # m*cz
    A[3, 4] = alphax                     # Ixx
    A[3, 5] = alphay - wx * wz           # Ixy
    A[3, 6] = alphaz + wx * wy           # Ixz
    A[3, 7] = -wy * wz                   # Iyy
    A[3, 8] = wy**2 - wz**2              # Iyz
    A[3, 9] = wy * wz                    # Izz

    # Row 5 (torque y): τ_y equation
    A[4, 1] = gz - az                    # m*cx
    A[4, 2] = 0                          # m*cy
    A[4, 3] = ax - gx                    # m*cz
    A[4, 4] = wx * wz                    # Ixx
    A[4, 5] = alphax + wy * wz           # Ixy
    A[4, 6] = wz**2 - wx**2              # Ixz
    A[4, 7] = alphay                     # Iyy
    A[4, 8] = alphaz - wx * wy           # Iyz
    A[4, 9] = -wx * wz                   # Izz

    # Row 6 (torque z): τ_z equation
    A[5, 1] = ay - gy                    # m*cx
    A[5, 2] = gx - ax                    # m*cy
    A[5, 3] = 0                          # m*cz
    A[5, 4] = -wx * wy                   # Ixx
    A[5, 5] = wx**2 - wy**2              # Ixy
    A[5, 6] = alphax - wy * wz           # Ixz
    A[5, 7] = wx * wy                    # Iyy
    A[5, 8] = alphay + wx * wz           # Iyz
    A[5, 9] = alphaz                     # Izz

    return A
