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
        payload: Payload object (CuboidPayload or CylinderPayload) with
            mass, inertia_tensor, and com_offset attributes.
        robot_prim_path: USD path to the robot articulation.
    """
    from omni.usd import get_context
    from models.payloads.cylinder import CylinderPayload
    from models.payloads.two_stage_cylinder import TwoStageCylinderPayload

    stage = get_context().get_stage()

    # Resolve attach link
    link_path = f"{robot_prim_path}/tool0"
    link_prim = stage.GetPrimAtPath(link_path)
    if not link_prim or not link_prim.IsValid():
        link_path = f"{robot_prim_path}/wrist_3_link"

    # Create geometry based on payload type
    payload_path = f"{link_path}/payload"

    if isinstance(payload, TwoStageCylinderPayload):
        # Create two-stage cylinder (stacked cylinders)
        # Lower cylinder (small, attached to tool0)
        lower_path = f"{link_path}/payload_lower"
        geom_lower = UsdGeom.Cylinder.Define(stage, lower_path)
        geom_lower.GetAxisAttr().Set("Z")
        geom_lower.GetRadiusAttr().Set(payload.lower_radius)
        geom_lower.GetHeightAttr().Set(payload.lower_height)
        # Position: COM at +height/2
        z_lower = payload.lower_height / 2
        geom_lower.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_lower))

        # Upper cylinder (large, on top of lower)
        upper_path = f"{link_path}/payload_upper"
        geom_upper = UsdGeom.Cylinder.Define(stage, upper_path)
        geom_upper.GetAxisAttr().Set("Z")
        geom_upper.GetRadiusAttr().Set(payload.upper_radius)
        geom_upper.GetHeightAttr().Set(payload.upper_height)
        # Position: COM at lower_height + height/2
        z_upper = payload.lower_height + payload.upper_height / 2
        geom_upper.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z_upper))

        # Apply mass properties to both geometries
        # Lower cylinder
        mass_api_lower = UsdPhysics.MassAPI.Apply(geom_lower.GetPrim())
        mass_api_lower.CreateMassAttr().Set(payload.lower_mass)
        mass_api_lower.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))
        I_lower = payload._cylinder_inertia_at_com(
            payload.lower_radius, payload.lower_height, payload.lower_mass
        )
        mass_api_lower.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(float(I_lower[0, 0]), float(I_lower[1, 1]), float(I_lower[2, 2]))
        )

        # Upper cylinder
        mass_api_upper = UsdPhysics.MassAPI.Apply(geom_upper.GetPrim())
        mass_api_upper.CreateMassAttr().Set(payload.upper_mass)
        mass_api_upper.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))
        I_upper = payload._cylinder_inertia_at_com(
            payload.upper_radius, payload.upper_height, payload.upper_mass
        )
        mass_api_upper.CreateDiagonalInertiaAttr().Set(
            Gf.Vec3f(float(I_upper[0, 0]), float(I_upper[1, 1]), float(I_upper[2, 2]))
        )

        # Collision + visual for both
        UsdPhysics.CollisionAPI.Apply(geom_lower.GetPrim())
        UsdPhysics.CollisionAPI.Apply(geom_upper.GetPrim())
        geom_lower.GetDisplayColorAttr().Set([(0.7, 0.7, 0.75)])  # Slightly darker for lower
        geom_upper.GetDisplayColorAttr().Set([(0.8, 0.8, 0.85)])

        shape_info = (
            f"2-stage: lower(r={payload.lower_radius*100:.0f}cm,h={payload.lower_height*100:.0f}cm), "
            f"upper(r={payload.upper_radius*100:.0f}cm,h={payload.upper_height*100:.0f}cm)"
        )

        # Use combined mass for info
        print(f"[INFO] Payload embedded in {link_path} "
              f"({shape_info}, total_mass={payload.mass:.2f} kg, combined_COM={payload.com_offset})")
        return  # Early return for two-stage

    elif isinstance(payload, CylinderPayload):
        # Create cylinder (bottom face attached to tool0, extends upward along +Z)
        geom = UsdGeom.Cylinder.Define(stage, payload_path)
        # USD Cylinder: axis="Z" means height extends along Z-axis
        geom.GetAxisAttr().Set("Z")
        geom.GetRadiusAttr().Set(payload.radius)
        geom.GetHeightAttr().Set(payload.height)
        # COM offset: bottom face at tool0, COM at +height/2 (upward)
        geom.AddTranslateOp().Set(Gf.Vec3d(*payload.com_offset))
        shape_info = f"radius={payload.radius*100:.0f}cm, height={payload.height*100:.0f}cm"
    else:
        # Create cuboid (default)
        geom = UsdGeom.Cube.Define(stage, payload_path)
        geom.AddTranslateOp().Set(Gf.Vec3d(*payload.com_offset))
        scale = Gf.Vec3f(
            payload.width / 2,
            payload.height / 2,
            payload.depth / 2,
        )
        geom.AddScaleOp().Set(scale)
        shape_info = f"{payload.width*100:.0f}x{payload.height*100:.0f}x{payload.depth*100:.0f}cm"

    # Mass (contributes to parent rigid body — no RigidBodyAPI here)
    mass_api = UsdPhysics.MassAPI.Apply(geom.GetPrim())
    mass_api.CreateMassAttr().Set(payload.mass)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(0, 0, 0))

    I = payload.inertia_tensor
    mass_api.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(float(I[0, 0]), float(I[1, 1]), float(I[2, 2]))
    )

    # Collision + visual
    UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
    geom.GetDisplayColorAttr().Set([(0.8, 0.8, 0.85)])

    print(f"[INFO] Payload embedded in {link_path} "
          f"({shape_info}, mass={payload.mass:.2f} kg, offset={payload.com_offset})")
