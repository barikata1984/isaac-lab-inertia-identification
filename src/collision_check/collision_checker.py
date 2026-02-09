"""FK-based geometric collision checker for UR5e.

Uses simplified collision geometry (spheres at joint positions)
for fast evaluation during optimization. More accurate collision
detection (Isaac Sim physics) is used for post-optimization validation.

The following collision types are checked:
1. Robot-ground: all joint positions z > ground_z_min
2. Payload-ground: payload bottom z > ground_z_min
3. Robot self-collision: distance between non-adjacent link centers
4. Payload-robot: distance between payload and non-adjacent links
"""

from dataclasses import dataclass, field

import numpy as np
import pinocchio as pin


@dataclass
class CollisionConfig:
    """Configuration for FK-based geometric collision checking.

    Attributes:
        ground_z_min: Minimum z-coordinate for all link positions [m].
        self_collision_min_dist: Minimum distance between non-adjacent
            link centers beyond the sum of their radii [m].
        payload_half_extents: Payload cuboid half-extents (3,) [m].
        payload_offset: Offset from tool0 to payload center (3,) [m].
        link_radii: Approximate sphere radii for each of the 6 links [m].
        enabled: Whether collision checking is active.
    """

    ground_z_min: float = 0.01
    self_collision_min_dist: float = 0.02
    payload_half_extents: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.075, 0.10])
    )
    payload_offset: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.10])
    )
    link_radii: np.ndarray = field(
        default_factory=lambda: np.array([0.06, 0.05, 0.045, 0.035, 0.035, 0.03])
    )
    enabled: bool = True


class CollisionChecker:
    """FK-based geometric collision checker for UR5e.

    Uses Pinocchio forward kinematics to compute joint positions
    and performs sphere-based distance checks for collision detection.
    """

    # Non-adjacent link pairs to check for self-collision (0-indexed).
    # Adjacent pairs (0,1), (1,2), (2,3), (3,4), (4,5) are excluded.
    SELF_COLLISION_PAIRS = [
        (0, 2), (0, 3), (0, 4), (0, 5),
        (1, 3), (1, 4), (1, 5),
        (2, 4), (2, 5),
        (3, 5),
    ]

    # Links NOT adjacent to tool0 (joint 5) for payload-link checks.
    PAYLOAD_CHECK_LINKS = [0, 1, 2, 3]

    def __init__(
        self,
        model: pin.Model,
        config: CollisionConfig | None = None,
    ):
        """Initialize collision checker.

        Args:
            model: Pinocchio model.
            config: Collision configuration. Uses defaults if None.
        """
        self.model = model
        self.data = model.createData()
        self.config = config or CollisionConfig()

        # Find tool0 frame ID
        self._tool0_id = model.getFrameId("tool0")

    def _get_joint_positions(self, q: np.ndarray) -> list[np.ndarray]:
        """Compute 3D positions of all joints via FK.

        Args:
            q: Joint configuration (6,).

        Returns:
            List of joint position vectors (3,), one per joint (1-indexed
            in Pinocchio, but returned 0-indexed here for 6 joints).
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        pin.forwardKinematics(self.model, self.data, q)

        positions = []
        for i in range(1, self.model.njoints):
            positions.append(self.data.oMi[i].translation.copy())

        return positions

    def _get_tool0_pose(
        self,
        q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute tool0 position and rotation.

        Assumes forwardKinematics has already been called.

        Args:
            q: Joint configuration (used if FK not yet computed).

        Returns:
            Tuple of (position (3,), rotation (3, 3)).
        """
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self._tool0_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def _payload_bottom_z(
        self,
        tool0_pos: np.ndarray,
        tool0_rot: np.ndarray,
    ) -> float:
        """Compute the lowest z-coordinate of the payload cuboid.

        Args:
            tool0_pos: Tool0 position in world frame (3,).
            tool0_rot: Tool0 rotation matrix (3, 3).

        Returns:
            Minimum z-coordinate of payload corners.
        """
        offset_world = tool0_rot @ self.config.payload_offset
        center = tool0_pos + offset_world

        he = self.config.payload_half_extents
        corners_local = np.array([
            [s0 * he[0], s1 * he[1], s2 * he[2]]
            for s0 in (-1, 1) for s1 in (-1, 1) for s2 in (-1, 1)
        ])
        corners_world = (tool0_rot @ corners_local.T).T + center
        return float(np.min(corners_world[:, 2]))

    def check_single_config(self, q: np.ndarray) -> float:
        """Compute minimum clearance for a single configuration.

        Positive = collision-free, negative = collision.

        Args:
            q: Joint configuration (6,).

        Returns:
            Minimum clearance [m].
        """
        q = np.asarray(q, dtype=np.float64).ravel()

        # Compute FK
        positions = self._get_joint_positions(q)
        tool0_pos, tool0_rot = self._get_tool0_pose(q)

        min_clearance = float("inf")

        # 1. Robot-ground: all joint z > ground_z_min
        for pos in positions:
            clearance = pos[2] - self.config.ground_z_min
            min_clearance = min(min_clearance, clearance)

        # 2. Self-collision: non-adjacent link distance check
        radii = self.config.link_radii
        for i, j in self.SELF_COLLISION_PAIRS:
            dist = np.linalg.norm(positions[i] - positions[j])
            margin = dist - (
                radii[i] + radii[j] + self.config.self_collision_min_dist
            )
            min_clearance = min(min_clearance, margin)

        # 3. Payload-ground
        bottom_z = self._payload_bottom_z(tool0_pos, tool0_rot)
        min_clearance = min(min_clearance, bottom_z - self.config.ground_z_min)

        # 4. Payload-robot links (non-adjacent to tool0)
        offset_world = tool0_rot @ self.config.payload_offset
        payload_center = tool0_pos + offset_world
        payload_radius = float(np.linalg.norm(self.config.payload_half_extents))

        for link_idx in self.PAYLOAD_CHECK_LINKS:
            dist = np.linalg.norm(payload_center - positions[link_idx])
            margin = dist - (
                payload_radius + radii[link_idx]
                + self.config.self_collision_min_dist
            )
            min_clearance = min(min_clearance, margin)

        return min_clearance

    def compute_min_clearance(
        self,
        q_trajectory: np.ndarray,
        subsample_factor: int = 5,
    ) -> float:
        """Compute minimum clearance across an entire trajectory.

        Args:
            q_trajectory: Joint positions (N, 6).
            subsample_factor: Check every N-th configuration for speed.

        Returns:
            Minimum clearance [m] across all checked configurations.
        """
        min_clearance = float("inf")

        for i in range(0, len(q_trajectory), subsample_factor):
            clearance = self.check_single_config(q_trajectory[i])
            min_clearance = min(min_clearance, clearance)

            # Early termination if deeply in collision
            if min_clearance < -0.1:
                return min_clearance

        return min_clearance
