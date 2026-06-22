"""f1tenth_stack bringup WITHOUT the stock gap_driver.

Identical to f1tenth_stack/launch/bringup_launch.py (joy/teleop, VESC driver+odom,
urg lidar, ackermann_mux, static base_link->laser TF) but it does NOT start the stock
gap_driver_node, so our reactive_explorer is the sole publisher on /drive.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("f1tenth_stack")
    joy_teleop_config = os.path.join(share, "config", "joy_teleop.yaml")
    vesc_config = os.path.join(share, "config", "vesc.yaml")
    sensors_config = os.path.join(share, "config", "sensors.yaml")
    mux_config = os.path.join(share, "config", "mux.yaml")

    ld = LaunchDescription([
        DeclareLaunchArgument("joy_teleop_config", default_value=joy_teleop_config),
        DeclareLaunchArgument("vesc_config", default_value=vesc_config),
        DeclareLaunchArgument("sensors_config", default_value=sensors_config),
        DeclareLaunchArgument("mux_config", default_value=mux_config),
    ])

    ld.add_action(Node(
        package="joy", executable="joy_node", name="joy",
        parameters=[LaunchConfiguration("joy_teleop_config")],
    ))
    ld.add_action(Node(
        package="joy_teleop", executable="joy_teleop", name="joy_teleop",
        parameters=[LaunchConfiguration("joy_teleop_config")],
    ))
    ld.add_action(Node(
        package="vesc_ackermann", executable="ackermann_to_vesc_node",
        name="ackermann_to_vesc_node", parameters=[LaunchConfiguration("vesc_config")],
    ))
    ld.add_action(Node(
        package="vesc_ackermann", executable="vesc_to_odom_node",
        name="vesc_to_odom_node", parameters=[LaunchConfiguration("vesc_config")],
    ))
    ld.add_action(Node(
        package="vesc_driver", executable="vesc_driver_node",
        name="vesc_driver_node", parameters=[LaunchConfiguration("vesc_config")],
    ))
    ld.add_action(Node(
        package="urg_node", executable="urg_node_driver", name="urg_node",
        parameters=[LaunchConfiguration("sensors_config")],
    ))
    ld.add_action(Node(
        package="ackermann_mux", executable="ackermann_mux", name="ackermann_mux",
        parameters=[LaunchConfiguration("mux_config")],
        remappings=[("ackermann_cmd_out", "ackermann_drive")],
    ))
    ld.add_action(Node(
        package="tf2_ros", executable="static_transform_publisher",
        name="static_baselink_to_laser",
        arguments=["0.27", "0.0", "0.11", "0.0", "0.0", "0.0", "base_link", "laser"],
    ))
    # NOTE: stock gap_driver_node intentionally omitted.
    return ld
