"""UR5e scene setup utilities for Isaac Sim.

All functions in this module require AppLauncher to be initialized.
"""

import numpy as np

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from pxr import UsdGeom, UsdPhysics, Gf

from models.robots.ur.ur5e import Q0_IDENTIFICATION, JOINT_NAMES


def make_ur5e_cfg(
    q0: np.ndarray | None = None,
    enable_self_collisions: bool = True,
) -> ArticulationCfg:
    """Create UR5e ArticulationCfg with configurable initial pose.

    Args:
        q0: Initial joint positions (6,). Defaults to Q0_IDENTIFICATION.
        enable_self_collisions: Enable self-collision detection in physics.
    """
    if q0 is None:
        q0 = Q0_IDENTIFICATION

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=enable_self_collisions,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            activate_contact_sensors=False,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                name: float(q0[i]) for i, name in enumerate(JOINT_NAMES)
            },
        ),
        actuators={
            "shoulder": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_.*"],
                stiffness=800.0,
                damping=40.0,
            ),
            "elbow": ImplicitActuatorCfg(
                joint_names_expr=["elbow_joint"],
                stiffness=800.0,
                damping=40.0,
            ),
            "wrist": ImplicitActuatorCfg(
                joint_names_expr=["wrist_.*"],
                stiffness=800.0,
                damping=40.0,
            ),
        },
    )


def design_scene(
    ur5e_cfg: ArticulationCfg | None = None,
) -> tuple[dict, list[list[float]]]:
    """Create ground, light, and UR5e robot.

    Args:
        ur5e_cfg: UR5e configuration. Uses make_ur5e_cfg() defaults if None.

    Returns:
        (scene_entities dict, origins list)
    """
    if ur5e_cfg is None:
        ur5e_cfg = make_ur5e_cfg()

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    origins = [[0.0, 0.0, 0.0]]
    prim_utils.create_prim("/World/Origin", "Xform", translation=origins[0])

    ur5e_cfg = ur5e_cfg.copy()
    ur5e_cfg.prim_path = "/World/Origin/Robot"
    ur5e = Articulation(cfg=ur5e_cfg)

    return {"ur5e": ur5e}, origins


def create_and_attach_payload(
    payload,
    robot_prim_path: str = "/World/Origin/Robot",
) -> None:
    """Embed payload as a child prim of wrist_3_link.

    The payload becomes part of the link's rigid body (no separate
    fixed joint needed).  Must be called BEFORE sim.reset() so that
    the physics engine initialises with the payload already attached.

    Args:
        payload: Payload object with width, height, depth, mass,
            inertia_tensor, and com_offset attributes.
        robot_prim_path: USD path to the robot articulation.
    """
    from omni.usd import get_context
    stage = get_context().get_stage()

    # Resolve attach link
    link_path = f"{robot_prim_path}/tool0"
    link_prim = stage.GetPrimAtPath(link_path)
    if not link_prim or not link_prim.IsValid():
        link_path = f"{robot_prim_path}/wrist_3_link"

    # Create cube as a CHILD of the link (rigid attachment, no joint)
    payload_path = f"{link_path}/payload"
    cube = UsdGeom.Cube.Define(stage, payload_path)

    # Local offset from link origin
    cube.AddTranslateOp().Set(Gf.Vec3d(*payload.com_offset))
    scale = Gf.Vec3f(
        payload.width / 2,
        payload.height / 2,
        payload.depth / 2,
    )
    cube.AddScaleOp().Set(scale)

    # Mass (contributes to parent rigid body — no RigidBodyAPI here)
    mass_api = UsdPhysics.MassAPI.Apply(cube.GetPrim())
    mass_api.CreateMassAttr().Set(payload.mass)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))

    I = payload.inertia_tensor
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(float(I[0, 0]), float(I[1, 1]), float(I[2, 2]))
    )

    # Collision + visual
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    cube.GetDisplayColorAttr().Set([(0.8, 0.8, 0.85)])

    print(f"[INFO] Payload embedded in {link_path} "
          f"(mass={payload.mass:.2f} kg, offset={payload.com_offset})")
