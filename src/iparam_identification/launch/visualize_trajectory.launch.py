"""Launch file for trajectory visualization in Isaac Sim GUI."""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_prefix


def generate_launch_description():
    """Generate launch description for trajectory visualization."""

    pkg_prefix = get_package_prefix('iparam_identification')
    viz_script = os.path.join(
        pkg_prefix, 'lib', 'iparam_identification', 'visualize_trajectory.sh',
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'trajectory',
            default_value='',
            description='Path to optimized_trajectory.json (empty = default)',
        ),
        DeclareLaunchArgument(
            'speed',
            default_value='1.0',
            description='Playback speed factor',
        ),
        DeclareLaunchArgument(
            'loop',
            default_value='false',
            description='Loop trajectory playback (true/false)',
        ),

        ExecuteProcess(
            cmd=[
                viz_script,
                '--speed', LaunchConfiguration('speed'),
            ],
            output='screen',
        ),
    ])
