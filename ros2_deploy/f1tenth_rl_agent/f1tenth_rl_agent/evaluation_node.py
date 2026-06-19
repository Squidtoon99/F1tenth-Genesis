"""evaluation_node: monitor the closed loop and report success metrics.

Uses the same Frenet projection as the observation builder to track lap progress,
lateral error, out-of-bounds and stuck conditions. Publishes ``/rl/metrics`` and,
when ``auto_reset`` is enabled, resets the car via ``/initialpose`` on episode end.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.eval_logic import EpisodeMonitor
from f1tenth_rl_agent.obs_core import ObservationBuilder


def _latched_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        history=QoSHistoryPolicy.KEEP_LAST,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    )


class EvaluationNode(Node):
    def __init__(self, **kwargs):
        super().__init__("evaluation", **kwargs)
        self.declare_parameter("oob_margin_m", 0.0)
        self.declare_parameter("stuck_speed_mps", 0.2)
        self.declare_parameter("stuck_timeout_s", 3.0)
        self.declare_parameter("reset_x", 0.0)
        self.declare_parameter("reset_y", 2.0)
        self.declare_parameter("reset_yaw", -0.9)
        self.declare_parameter("auto_reset", True)

        gp = self.get_parameter
        self.reset_x = gp("reset_x").get_parameter_value().double_value
        self.reset_y = gp("reset_y").get_parameter_value().double_value
        self.reset_yaw = gp("reset_yaw").get_parameter_value().double_value
        self.auto_reset = gp("auto_reset").get_parameter_value().bool_value

        self.monitor = EpisodeMonitor(
            oob_margin_m=gp("oob_margin_m").get_parameter_value().double_value,
            stuck_speed_mps=gp("stuck_speed_mps").get_parameter_value().double_value,
            stuck_timeout_s=gp("stuck_timeout_s").get_parameter_value().double_value,
        )

        self.builder: ObservationBuilder | None = None
        self._centerline = None
        self._w_left = None
        self._w_right = None

        latched = _latched_qos()
        self.create_subscription(Path, ifc.TOPIC_TRACK_CENTERLINE, self._on_centerline, latched)
        self.create_subscription(
            Float32MultiArray, ifc.TOPIC_TRACK_WIDTHS, self._on_widths, latched
        )
        self.create_subscription(Odometry, ifc.TOPIC_ODOM, self._on_odom, 10)

        self.metrics_pub = self.create_publisher(Float32MultiArray, ifc.TOPIC_METRICS, 10)
        self.reset_pub = self.create_publisher(
            PoseWithCovarianceStamped, ifc.TOPIC_INITIALPOSE, 1
        )

    def _on_centerline(self, msg: Path):
        self._centerline = np.array(
            [[p.pose.position.x, p.pose.position.y] for p in msg.poses], dtype=np.float32
        )
        self._maybe_build()

    def _on_widths(self, msg: Float32MultiArray):
        data = np.asarray(msg.data, dtype=np.float32)
        self._w_left = data[0::2].copy()
        self._w_right = data[1::2].copy()
        self._maybe_build()

    def _maybe_build(self):
        if self.builder is not None:
            return
        if self._centerline is None or self._w_left is None or self._w_right is None:
            return
        if len(self._w_left) != len(self._centerline):
            return
        self.builder = ObservationBuilder(
            centerline=self._centerline,
            w_tr_left=self._w_left,
            w_tr_right=self._w_right,
            obs_cfg={"num_obs": ifc.NUM_OBS},
            device=torch.device("cpu"),
        )
        self.get_logger().info("evaluation monitor ready")

    def _on_odom(self, msg: Odometry):
        if self.builder is None:
            return
        pos = msg.pose.pose.position
        base_pos = torch.tensor([[pos.x, pos.y, pos.z]], dtype=torch.float32)
        frenet_state, boundary_state = self.builder.frenet(base_pos)

        s = float(frenet_state["s"][0])
        track_len = float(frenet_state["L"])
        ey = float(boundary_state["ey"][0])
        w_l = float(boundary_state["w_l_s"][0])
        w_r = float(boundary_state["w_r_s"][0])
        speed = math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        event = self.monitor.update(s, track_len, ey, w_l, w_r, speed, t)

        metrics = Float32MultiArray()
        metrics.data = [
            float(event.lap_count),
            float(self.monitor.last_lap_time or 0.0),
            float(event.max_progress),
            float(ey),
            1.0 if event.oob else 0.0,
            float(speed),
        ]
        self.metrics_pub.publish(metrics)

        if event.lap_completed:
            self.get_logger().info(
                f"LAP {event.lap_count} completed in {event.lap_time:.2f}s"
            )
        if event.oob:
            self.get_logger().warn(f"OUT OF BOUNDS (ey={ey:.2f})")
        if event.stuck:
            self.get_logger().warn("Car stuck")

        if (event.oob or event.stuck) and self.auto_reset:
            self._reset_car()
            self.monitor.reset(t)

    def _reset_car(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = ifc.FRAME_MAP
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = self.reset_x
        msg.pose.pose.position.y = self.reset_y
        msg.pose.pose.orientation.z = math.sin(self.reset_yaw / 2)
        msg.pose.pose.orientation.w = math.cos(self.reset_yaw / 2)
        self.reset_pub.publish(msg)
        self.get_logger().info("Published /initialpose reset")


def main(args=None):
    rclpy.init(args=args)
    node = EvaluationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
