"""Publish centerline + left/right boundaries for RViz alignment checks."""

from __future__ import annotations

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


def _load_track(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=np.float64)
    center = np.stack([raw["x_m"], raw["y_m"]], axis=1)
    w_left = np.asarray(raw["w_tr_left_m"], dtype=np.float64)
    w_right = np.asarray(raw["w_tr_right_m"], dtype=np.float64)
    return center, w_left, w_right


def _boundaries(
    center: np.ndarray, w_left: np.ndarray, w_right: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = center.shape[0]
    tang = np.zeros_like(center)
    for i in range(n):
        tang[i] = center[(i + 1) % n] - center[i - 1]
    tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
    normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    left = center + normal * w_left[:, None]
    right = center - normal * w_right[:, None]
    return left, right


def _line_marker(
    marker_id: int,
    frame_id: str,
    points: np.ndarray,
    color: tuple[float, float, float, float],
    scale: float,
    ns: str,
) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = marker_id
    m.type = Marker.LINE_STRIP
    m.action = Marker.ADD
    m.scale.x = scale
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.pose.orientation.w = 1.0
    for x, y in points:
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = 0.05
        m.points.append(p)
    # Close the loop for readability in RViz.
    if len(m.points) > 1:
        m.points.append(m.points[0])
    return m


def _point_marker(
    marker_id: int,
    frame_id: str,
    x: float,
    y: float,
    color: tuple[float, float, float, float],
    scale: float,
    ns: str,
) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = marker_id
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.scale.x = m.scale.y = m.scale.z = scale
    m.color.r, m.color.g, m.color.b, m.color.a = color
    m.pose.position.x = x
    m.pose.position.y = y
    m.pose.position.z = 0.08
    m.pose.orientation.w = 1.0
    return m


class TrackVizNode(Node):
    def __init__(self) -> None:
        super().__init__("track_viz")
        self.declare_parameter("track_csv", "")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("publish_hz", 1.0)
        self.declare_parameter("line_width", 0.06)

        csv_path = str(self.get_parameter("track_csv").value)
        if not csv_path:
            raise RuntimeError("track_csv parameter is required")

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._line_width = float(self.get_parameter("line_width").value)
        center, w_left, w_right = _load_track(csv_path)
        self._left, self._right = _boundaries(center, w_left, w_right)
        self._center = center

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub = self.create_publisher(MarkerArray, "/track/viz/markers", qos)
        hz = max(float(self.get_parameter("publish_hz").value), 0.1)
        self.create_timer(1.0 / hz, self._publish)
        self.get_logger().info(
            f"track viz: {center.shape[0]} pts from {csv_path} frame={self._frame_id}"
        )
        self._publish()

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()
        specs = (
            ("centerline", self._center, (0.1, 0.45, 1.0, 1.0), 0),
            ("left_boundary", self._left, (0.1, 0.85, 0.2, 1.0), 1),
            ("right_boundary", self._right, (0.95, 0.15, 0.55, 1.0), 2),
        )
        for ns, pts, color, mid in specs:
            m = _line_marker(mid, self._frame_id, pts, color, self._line_width, ns)
            m.header.stamp = stamp
            arr.markers.append(m)

        start = _point_marker(3, self._frame_id, self._center[0, 0], self._center[0, 1],
                              (0.2, 1.0, 0.2, 1.0), 0.25, "start")
        start.header.stamp = stamp
        arr.markers.append(start)
        self._pub.publish(arr)


def main() -> None:
    rclpy.init()
    node = TrackVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
