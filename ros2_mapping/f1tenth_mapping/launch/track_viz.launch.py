"""RViz debug: centerline + left/right boundaries in the map frame."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("f1tenth_mapping")
    default_csv = os.environ.get(
        "TRACK_CSV", "/home/shereef/maps/f1tenth_map_centerline.csv"
    )

    return LaunchDescription([
        DeclareLaunchArgument("track_csv", default_value=default_csv),
        DeclareLaunchArgument("frame_id", default_value="map"),
        Node(
            package="f1tenth_mapping",
            executable="track_viz",
            name="track_viz",
            output="screen",
            parameters=[{
                "track_csv": LaunchConfiguration("track_csv"),
                "frame_id": LaunchConfiguration("frame_id"),
                "publish_hz": 1.0,
                "line_width": 0.06,
            }],
        ),
    ])
