"""Launch the open-loop maneuver publisher for an on-car calibration run.

Run this ALONGSIDE ``bringup_vehicle.launch.py`` (which provides the ``drive``
node that maps actions to ``/drive`` with the staged ``speed_limit_mps`` cap and
watchdog). Do NOT launch ``policy_inference`` at the same time, or two publishers
will fight over ``/rl/action``.

The node is executed directly from the repo checkout (it is not part of a built
ament package), so pass the absolute path to the Python executable on the car if
it differs from ``python3``.

Example:
    ros2 launch vehicle_calibration/ros/launch/profile_maneuvers.launch.py \\
        maneuver_yaml:=/abs/path/vehicle_calibration/maneuvers/carpet_profile.yaml
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_THIS = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(os.path.dirname(_THIS))
_DEFAULT_YAML = os.path.join(_PKG, "maneuvers", "carpet_profile.yaml")


def generate_launch_description():
    maneuver_yaml = LaunchConfiguration("maneuver_yaml")
    start_delay_s = LaunchConfiguration("start_delay_s")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "maneuver_yaml",
                default_value=_DEFAULT_YAML,
                description="Shared maneuver schedule (same file Genesis profiles).",
            ),
            DeclareLaunchArgument(
                "start_delay_s",
                default_value="3.0",
                description="Seconds to wait before the first action (arm teleop first).",
            ),
            Node(
                executable="python3",
                arguments=[
                    os.path.join(_THIS, "..", "profile_maneuver_node.py"),
                ],
                name="profile_maneuver",
                output="screen",
                parameters=[
                    {
                        "maneuver_yaml": maneuver_yaml,
                        "start_delay_s": start_delay_s,
                    }
                ],
            ),
        ]
    )
