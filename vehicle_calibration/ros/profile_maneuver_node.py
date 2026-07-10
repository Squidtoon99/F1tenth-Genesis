"""profile_maneuver_node: open-loop maneuver action publisher for the real car.

Plays the SAME ``carpet_profile.yaml`` schedule used by the Genesis profiler,
publishing normalized ``[throttle, steer]`` on ``/rl/action`` at the control rate.
It bypasses ``policy_inference`` entirely (no checkpoint needed) and lets the
existing ``drive`` node apply the action -> Ackermann mapping plus the staged
``speed_limit_mps`` cap and watchdog.

It also publishes the current maneuver id and segment role on ``/calib/maneuver``
and ``/calib/role`` so ``parse_bag.py`` can label each recorded sample. On
completion (or shutdown) it latches a zero action so the car stops.

Run alongside ``bringup_vehicle.launch.py`` (which provides the ``drive`` node),
e.g. via ``vehicle_calibration/ros/launch/profile_maneuvers.launch.py``.
"""

from __future__ import annotations

import os
import sys

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray, String

# Allow running both as an installed entry point and straight from the repo.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from vehicle_calibration.maneuvers import load_schedule  # noqa: E402

DEFAULT_YAML = os.path.join(
    _REPO, "vehicle_calibration", "maneuvers", "carpet_profile.yaml"
)


class ProfileManeuverNode(Node):
    def __init__(self) -> None:
        super().__init__("profile_maneuver")
        self.declare_parameter("maneuver_yaml", DEFAULT_YAML)
        self.declare_parameter("action_topic", "/rl/action")
        self.declare_parameter("start_delay_s", 3.0)

        yaml_path = self.get_parameter("maneuver_yaml").value
        action_topic = self.get_parameter("action_topic").value
        self.start_delay_s = float(self.get_parameter("start_delay_s").value)

        self.schedule = load_schedule(yaml_path)
        self.control_hz = self.schedule.control_hz

        # Flatten the schedule into a per-control-step action timeline.
        self._steps: list[tuple[str, str, float, float]] = []
        for man in self.schedule.maneuvers:
            for seg in man.segments:
                n = round(seg.duration_s * self.control_hz)
                for _ in range(n):
                    self._steps.append((man.id, seg.role, seg.throttle, seg.steer))
        self._idx = 0
        self._started = False

        self.action_pub = self.create_publisher(Float32MultiArray, action_topic, 10)
        self.maneuver_pub = self.create_publisher(String, "/calib/maneuver", 10)
        self.role_pub = self.create_publisher(String, "/calib/role", 10)

        self.get_logger().info(
            f"profile_maneuver: {len(self.schedule.maneuvers)} maneuvers, "
            f"{len(self._steps)} steps @ {self.control_hz} Hz, "
            f"starting in {self.start_delay_s:.1f}s"
        )
        self._start_timer = self.create_timer(self.start_delay_s, self._begin)

    def _begin(self) -> None:
        self._start_timer.cancel()
        self._started = True
        self.timer = self.create_timer(1.0 / self.control_hz, self._tick)

    def _publish(self, throttle: float, steer: float, man: str, role: str) -> None:
        self.action_pub.publish(Float32MultiArray(data=[float(throttle), float(steer)]))
        self.maneuver_pub.publish(String(data=man))
        self.role_pub.publish(String(data=role))

    def _tick(self) -> None:
        if self._idx >= len(self._steps):
            self._publish(0.0, 0.0, "DONE", "done")
            self.get_logger().info("maneuver schedule complete; commanding stop")
            self.timer.cancel()
            return
        man, role, throttle, steer = self._steps[self._idx]
        self._publish(throttle, steer, man, role)
        self._idx += 1

    def stop(self) -> None:
        self._publish(0.0, 0.0, "ABORT", "done")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ProfileManeuverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
