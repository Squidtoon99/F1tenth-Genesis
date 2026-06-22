"""Bring up the lean on-car RL agent: 3 nodes.

Launches the two C++ vehicle nodes (vehicle_obs, drive) from f1tenth_rl_vehicle and
the Python policy_inference node from f1tenth_rl_agent. The f1tenth_stack (particle
filter + VESC + ackermann_mux) is launched separately and is not part of this file.

By default the track centerline is the IV_2026_SIM asset installed with
f1tenth_rl_agent; override track_csv for a surveyed real track.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    vehicle_share = get_package_share_directory("f1tenth_rl_vehicle")
    agent_share = get_package_share_directory("f1tenth_rl_agent")

    default_vehicle_params = os.path.join(vehicle_share, "config", "vehicle.yaml")
    default_agent_params = os.path.join(agent_share, "config", "agent.yaml")
    default_track_csv = os.path.join(
        agent_share, "assets", "IV_2026_SIM_centerline.csv"
    )

    params_file = LaunchConfiguration("params_file")
    agent_params_file = LaunchConfiguration("agent_params_file")
    checkpoint_path = LaunchConfiguration("checkpoint_path")
    track_csv = LaunchConfiguration("track_csv")
    enable_opponent = LaunchConfiguration("enable_opponent")
    # Bool literal for ROS parameter overrides (LaunchConfiguration resolves to a
    # string; pass the Python bool so YAML/ROS sees a real boolean).
    enable_opponent_bool = PythonExpression(["'", enable_opponent, "' == 'true'"])

    declare_params = DeclareLaunchArgument(
        "params_file",
        default_value=default_vehicle_params,
        description="vehicle_obs + drive parameter YAML.",
    )
    declare_agent_params = DeclareLaunchArgument(
        "agent_params_file",
        default_value=default_agent_params,
        description="policy_inference parameter YAML (from f1tenth_rl_agent).",
    )
    declare_ckpt = DeclareLaunchArgument(
        "checkpoint_path",
        default_value="/checkpoints/policy.pt",
        description="Trained policy .pt (must include obs_norm to match training).",
    )
    declare_track = DeclareLaunchArgument(
        "track_csv",
        default_value=default_track_csv,
        description="Centerline CSV in the same frame as the localization map.",
    )
    declare_opponent = DeclareLaunchArgument(
        "enable_opponent",
        default_value="false",
        description="Enable LiDAR opponent detection + the 387-dim 1v1 observation.",
    )

    nodes = [
        Node(
            package="f1tenth_rl_vehicle",
            executable="vehicle_obs",
            name="vehicle_obs",
            parameters=[
                params_file,
                {"track_csv": track_csv, "enable_opponent_obs": enable_opponent_bool},
            ],
            output="screen",
        ),
        Node(
            package="f1tenth_rl_agent",
            executable="policy_inference",
            name="policy_inference",
            parameters=[
                agent_params_file,
                {
                    "checkpoint_path": checkpoint_path,
                    "enable_opponent_obs": enable_opponent_bool,
                },
            ],
            output="screen",
        ),
        Node(
            package="f1tenth_rl_vehicle",
            executable="drive",
            name="drive",
            parameters=[params_file],
            output="screen",
        ),
        Node(
            package="f1tenth_rl_vehicle",
            executable="opponent_detector",
            name="opponent_detector",
            parameters=[params_file, {"track_csv": track_csv}],
            output="screen",
            condition=IfCondition(enable_opponent),
        ),
    ]

    return LaunchDescription(
        [
            declare_params,
            declare_agent_params,
            declare_ckpt,
            declare_track,
            declare_opponent,
            *nodes,
        ]
    )
