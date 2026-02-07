"""Isaac Sim GUI で UR5e ロボットをワールドにスポーンして表示するスクリプト。

Usage:
    isaac-python scripts/spawn_ur5e.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Spawn UR5e robot in Isaac Sim GUI.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


# UR5e configuration
UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e/ur5e.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5707,
            "elbow_joint": 1.5707,
            "wrist_1_joint": -1.5707,
            "wrist_2_joint": -1.5707,
            "wrist_3_joint": 0.0,
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


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene with a UR5e robot."""
    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Dome light
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Robot origin
    origins = [[0.0, 0.0, 0.0]]
    sim_utils.create_prim("/World/Origin", "Xform", translation=origins[0])

    # Spawn UR5e
    ur5e_cfg = UR5E_CFG.copy()
    ur5e_cfg.prim_path = "/World/Origin/Robot"
    ur5e = Articulation(cfg=ur5e_cfg)

    scene_entities = {"ur5e": ur5e}
    return scene_entities, origins


def run_simulator(sim: SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    robot = entities["ur5e"]
    sim_dt = sim.get_physics_dt()
    count = 0

    while simulation_app.is_running():
        # Reset every 1000 steps
        if count % 1000 == 0:
            count = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print("[INFO]: Resetting robot state...")

        # Hold position (zero effort — PD controller maintains joint positions)
        robot.write_data_to_sim()
        sim.step()
        count += 1
        robot.update(sim_dt)


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Camera positioned to view the robot
    sim.set_camera_view([1.5, 1.5, 1.0], [0.0, 0.0, 0.3])

    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()
    print("[INFO]: Setup complete. UR5e spawned.")

    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
