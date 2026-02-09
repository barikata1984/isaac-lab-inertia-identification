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

from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
import isaaclab.sim as sim_utils

from isaac_utils.bootstrap import setup_quit_handler
from isaac_utils.scene import make_ur5e_cfg, design_scene

# Register GUI close handlers
_quit_subs = setup_quit_handler()


def run_simulator(sim: SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    robot = entities["ur5e"]
    sim_dt = sim.get_physics_dt()
    count = 0

    # NOTE: GUI pause/play causes a rendering freeze after resume.
    # This is a known Isaac Sim 5.1 bug (https://github.com/isaac-sim/IsaacLab/issues/4279).
    # sim.step() blocks while paused and resumes automatically on play.
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

        # Set PD position targets to hold the default pose
        robot.set_joint_position_target(robot.data.default_joint_pos)
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

    ur5e_cfg = make_ur5e_cfg(enable_self_collisions=False)
    scene_entities, scene_origins = design_scene(ur5e_cfg)
    scene_origins = torch.tensor(scene_origins, device=sim.device)

    sim.reset()
    print("[INFO]: Setup complete. UR5e spawned.")

    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
