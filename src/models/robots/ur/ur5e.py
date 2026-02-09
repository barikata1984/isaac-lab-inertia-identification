"""UR5e robot model constants."""

import numpy as np

URDF_PATH = (
    "/isaac-sim/exts/isaacsim.robot_motion.motion_generation"
    "/motion_policy_configs/universal_robots/ur5e/ur5e.urdf"
)

Q0_IDENTIFICATION = np.array([
    np.pi / 2, -np.pi / 2, np.pi / 2,
    -np.pi / 2, -np.pi / 2, np.pi / 2,
])

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
