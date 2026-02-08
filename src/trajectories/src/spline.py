from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base_trajectory import BaseTrajectory, BaseTrajectoryConfig


@dataclass(kw_only=True)
class SplineTrajectoryConfig(BaseTrajectoryConfig):
    type: str = "septic"  # "quintic" or "septic"
    config: Path | None = None  # Path to YAML configuration file
    start_pos: list[float]  # Required, no default
    end_pos: list[float]  # Required, no default
    start_vel: list[float] | None = None
    end_vel: list[float] | None = None
    start_acc: list[float] | None = None
    end_acc: list[float] | None = None
    start_jerk: list[float] | None = None
    end_jerk: list[float] | None = None
    target_class: str = "SplineTrajectory"


class SplineTrajectory(BaseTrajectory):
    """Generates a polynomial trajectory (quintic or septic) for multiple joints.
    Ensures continuous position, velocity, and acceleration.
    """

    def __init__(
        self,
        cfg: SplineTrajectoryConfig,
        *args,
        **kwargs,
    ):
        """Initialize the trajectory generator.

        Args:
           cfg: Configuration object.
        """
        super().__init__(cfg, *args, **kwargs)

        self.start_pos = np.array(cfg.start_pos)
        self.end_pos = np.array(cfg.end_pos)
        self.num_joints = len(cfg.start_pos)
        self.type = cfg.type

        if len(cfg.end_pos) != self.num_joints:
            raise ValueError("Start and end positions must have the same length.")

        self.start_vel = np.array(cfg.start_vel) if cfg.start_vel is not None else np.zeros(self.num_joints)
        self.end_vel = np.array(cfg.end_vel) if cfg.end_vel is not None else np.zeros(self.num_joints)
        self.start_acc = np.array(cfg.start_acc) if cfg.start_acc is not None else np.zeros(self.num_joints)
        self.end_acc = np.array(cfg.end_acc) if cfg.end_acc is not None else np.zeros(self.num_joints)
        self.start_jerk = np.array(cfg.start_jerk) if cfg.start_jerk is not None else np.zeros(self.num_joints)
        self.end_jerk = np.array(cfg.end_jerk) if cfg.end_jerk is not None else np.zeros(self.num_joints)

        self.time_steps = int(self.duration * self.fps)
        self.time_array = np.linspace(0, self.duration, self.time_steps)

        # Pre-calculate coefficients based on type
        if self.type == "quintic":
            self.coeffs = self._calculate_quintic_coefficients()
        elif self.type == "septic":
            self.coeffs = self._calculate_septic_coefficients()
        else:
            raise ValueError(f"Unknown spline type: {self.type}")

    def _calculate_quintic_coefficients(self) -> np.ndarray:
        """Calculates the coefficients a0, a1, a2, a3, a4, a5 for the polynomial:
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        """
        T = self.duration
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T

        # System of equations for a quintic polynomial
        # q(0) = a0
        # v(0) = a1
        # a(0) = 2*a2
        # q(T) = a0 + a1*T + a2*T^2 + a3*T^3 + a4*T^4 + a5*T^5
        # v(T) = a1 + 2*a2*T + 3*a3*T^2 + 4*a4*T^3 + 5*a5*T^4
        # a(T) = 2*a2 + 6*a3*T + 12*a4*T^2 + 20*a5*T^3

        coeffs = np.zeros((self.num_joints, 6))

        for i in range(self.num_joints):
            q0 = self.start_pos[i]
            v0 = self.start_vel[i]
            acc0 = self.start_acc[i]
            q1 = self.end_pos[i]
            v1 = self.end_vel[i]
            acc1 = self.end_acc[i]

            a0 = q0
            a1 = v0
            a2 = acc0 / 2.0

            # Solve for a3, a4, a5
            # A * [a3, a4, a5]^T = B
            A = np.array(
                [
                    [T3, T4, T5],
                    [3 * T2, 4 * T3, 5 * T4],
                    [6 * T, 12 * T2, 20 * T3],
                ],
            )

            B = np.array(
                [
                    q1 - (a0 + a1 * T + a2 * T2),
                    v1 - (a1 + 2 * a2 * T),
                    acc1 - (2 * a2),
                ],
            )

            x = np.linalg.solve(A, B)

            coeffs[i, 0] = a0
            coeffs[i, 1] = a1
            coeffs[i, 2] = a2
            coeffs[i, 3] = x[0]
            coeffs[i, 4] = x[1]
            coeffs[i, 5] = x[2]

        return coeffs

    def _calculate_septic_coefficients(self) -> np.ndarray:
        """Calculates the coefficients a0...a7 for a septic (7th-order) polynomial.
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5 + a6*t^6 + a7*t^7
        Ensures continuous jerk at boundaries.
        """
        T = self.duration
        T2 = T * T
        T3 = T2 * T
        T4 = T3 * T
        T5 = T4 * T
        T6 = T5 * T
        T7 = T6 * T

        # q(0) = a0
        # v(0) = a1
        # a(0) = 2*a2
        # j(0) = 6*a3

        # q(T) = a0 + a1*T + a2*T2 + a3*T3 + a4*T4 + a5*T5 + a6*T6 + a7*T7
        # v(T) = a1 + 2a2T + 3a3T2 + 4a4T3 + 5a5T4 + 6a6T5 + 7a7T6
        # a(T) = 2a2 + 6a3T + 12a4T2 + 20a5T3 + 30a6T4 + 42a7T5
        # j(T) = 6a3 + 24a4T + 60a5T2 + 120a6T3 + 210a7T4

        coeffs = np.zeros((self.num_joints, 8))

        for i in range(self.num_joints):
            q0 = self.start_pos[i]
            v0 = self.start_vel[i]
            acc0 = self.start_acc[i]
            j0 = self.start_jerk[i]

            q1 = self.end_pos[i]
            v1 = self.end_vel[i]
            acc1 = self.end_acc[i]
            j1 = self.end_jerk[i]

            a0 = q0
            a1 = v0
            a2 = acc0 / 2.0
            a3 = j0 / 6.0

            # Solve for a4, a5, a6, a7
            # A * [a4, a5, a6, a7]^T = B
            A = np.array(
                [
                    [T4, T5, T6, T7],
                    [4 * T3, 5 * T4, 6 * T5, 7 * T6],
                    [12 * T2, 20 * T3, 30 * T4, 42 * T5],
                    [24 * T, 60 * T2, 120 * T3, 210 * T4],
                ],
            )

            known_q = a0 + a1 * T + a2 * T2 + a3 * T3
            known_v = a1 + 2 * a2 * T + 3 * a3 * T2
            known_a = 2 * a2 + 6 * a3 * T
            known_j = 6 * a3

            B = np.array(
                [
                    q1 - known_q,
                    v1 - known_v,
                    acc1 - known_a,
                    j1 - known_j,
                ],
            )

            x = np.linalg.solve(A, B)

            coeffs[i, 0] = a0
            coeffs[i, 1] = a1
            coeffs[i, 2] = a2
            coeffs[i, 3] = a3
            coeffs[i, 4] = x[0]
            coeffs[i, 5] = x[1]
            coeffs[i, 6] = x[2]
            coeffs[i, 7] = x[3]

        return coeffs

    def get_value(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate position, velocity, and acceleration at all time steps.

        Returns:
            pos: (N, num_joints) Position array
            vel: (N, num_joints) Velocity array
            acc: (N, num_joints) Acceleration array
        """
        time_arr = np.atleast_1d(self.time_array)
        num_steps = len(time_arr)

        pos = np.zeros((num_steps, self.num_joints))
        vel = np.zeros((num_steps, self.num_joints))
        acc = np.zeros((num_steps, self.num_joints))

        for t_idx, t in enumerate(time_arr):
            t2 = t * t
            t3 = t2 * t
            t4 = t3 * t
            t5 = t4 * t

            for j in range(self.num_joints):
                if self.type == "quintic":
                    a0, a1, a2, a3, a4, a5 = self.coeffs[j]
                    pos[t_idx, j] = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5
                    vel[t_idx, j] = a1 + 2 * a2 * t + 3 * a3 * t2 + 4 * a4 * t3 + 5 * a5 * t4
                    acc[t_idx, j] = 2 * a2 + 6 * a3 * t + 12 * a4 * t2 + 20 * a5 * t3
                elif self.type == "septic":
                    a0, a1, a2, a3, a4, a5, a6, a7 = self.coeffs[j]
                    t6 = t5 * t
                    t7 = t6 * t
                    pos[t_idx, j] = a0 + a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5 + a6 * t6 + a7 * t7
                    vel[t_idx, j] = (
                        a1 + 2 * a2 * t + 3 * a3 * t2 + 4 * a4 * t3 + 5 * a5 * t4 + 6 * a6 * t5 + 7 * a7 * t6
                    )
                    acc[t_idx, j] = 2 * a2 + 6 * a3 * t + 12 * a4 * t2 + 20 * a5 * t3 + 30 * a6 * t4 + 42 * a7 * t5

        return pos, vel, acc

    def _generate(self, show_plot: bool = False, plot_path: str | None = None, json_path: str | None = None):
        """Generates the trajectory.

        Returns:
            positions: (num_steps, num_joints)
            velocities: (num_steps, num_joints)
            accelerations: (num_steps, num_joints)
        """
        return self.get_value()


if __name__ == "__main__":
    # Example usage
    start_q = [
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    end_q = [
        0.2,
        1.4,
        0.6,
        3.141592653589793,  # 1π
        0.0,
        25.1327412287,  # 8
    ]

    duration = 5.0
    fps = 60.0

    cfg = SplineTrajectoryConfig(
        duration=duration,
        fps=fps,
        type="quintic",
        start_pos=start_q,
        end_pos=end_q,
    )

    traj = SplineTrajectory(cfg)

    traj.generate(show_plot=True, plot_path="debug/spline.png", json_path="configurations/trajectories/spline.json")
