"""policy_inference_node: run the trained actor on incoming observations.

Subscribes to ``/rl/observation`` (372-dim) and publishes ``/rl/action`` (2-dim,
in [-1, 1]). If no checkpoint is available it falls back to a randomly initialized
actor so the rest of the pipeline can still be exercised (logged as a warning).
"""

from __future__ import annotations

import time

import numpy as np
import rclpy
import torch
import torch.nn as nn
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.policy_model import SquashedGaussianMLPActor, load_actor


class PolicyInferenceNode(Node):
    def __init__(self, **kwargs):
        super().__init__("policy_inference", **kwargs)
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("state_dict_key", "actor")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("deterministic", True)
        self.declare_parameter("demo_throttle_floor", 0.0)

        gp = self.get_parameter
        checkpoint_path = gp("checkpoint_path").get_parameter_value().string_value
        state_dict_key = gp("state_dict_key").get_parameter_value().string_value
        device_str = gp("device").get_parameter_value().string_value
        self.deterministic = gp("deterministic").get_parameter_value().bool_value
        self.demo_throttle_floor = (
            gp("demo_throttle_floor").get_parameter_value().double_value
        )

        self.device = torch.device(device_str)
        self.actor, self._checkpoint_loaded = self._load_actor(
            checkpoint_path, state_dict_key
        )

        self.action_pub = self.create_publisher(Float32MultiArray, ifc.TOPIC_ACTION, 10)
        self.create_subscription(
            Float32MultiArray, ifc.TOPIC_OBSERVATION, self._on_obs, 10
        )
        self._infer_count = 0
        self._infer_time_accum = 0.0

    def _load_actor(self, checkpoint_path: str, state_dict_key: str):
        if checkpoint_path:
            try:
                actor = load_actor(
                    checkpoint_path=checkpoint_path,
                    obs_dim=ifc.NUM_OBS,
                    act_dim=ifc.NUM_ACTIONS,
                    hidden_sizes=ifc.HIDDEN_LAYERS,
                    act_limit=ifc.ACT_LIMIT,
                    state_dict_key=state_dict_key,
                    device=self.device,
                )
                self.get_logger().info(f"Loaded policy checkpoint: {checkpoint_path}")
                return actor, True
            except Exception as exc:  # noqa: BLE001
                self.get_logger().error(
                    f"Failed to load checkpoint '{checkpoint_path}': {exc}. "
                    "Falling back to random-init actor."
                )
        else:
            self.get_logger().warn(
                "No checkpoint_path provided; using random-init actor (plumbing only)."
            )
        actor = SquashedGaussianMLPActor(
            obs_dim=ifc.NUM_OBS,
            act_dim=ifc.NUM_ACTIONS,
            hidden_sizes=ifc.HIDDEN_LAYERS,
            activation=nn.ReLU,
            act_limit=ifc.ACT_LIMIT,
        ).to(self.device)
        actor.eval()
        return actor, False

    def _on_obs(self, msg: Float32MultiArray):
        if len(msg.data) != ifc.NUM_OBS:
            self.get_logger().warn(
                f"Observation length {len(msg.data)} != {ifc.NUM_OBS}; skipping"
            )
            return
        obs = torch.tensor([list(msg.data)], dtype=torch.float32, device=self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            action, _ = self.actor(
                obs, deterministic=self.deterministic, with_logprob=False
            )
        self._infer_time_accum += time.perf_counter() - t0
        self._infer_count += 1

        action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
        action_np = np.clip(action_np, -ifc.CLIP_ACTIONS, ifc.CLIP_ACTIONS)
        if not self._checkpoint_loaded and self.demo_throttle_floor > 0.0:
            action_np[0] = max(action_np[0], float(self.demo_throttle_floor))

        out = Float32MultiArray()
        out.data = action_np.tolist()
        self.action_pub.publish(out)

        if self._infer_count % 100 == 0:
            avg_ms = 1000.0 * self._infer_time_accum / self._infer_count
            self.get_logger().info(f"inference avg latency {avg_ms:.2f} ms")


def main(args=None):
    rclpy.init(args=args)
    node = PolicyInferenceNode()
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
