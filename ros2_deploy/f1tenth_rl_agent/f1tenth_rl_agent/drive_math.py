"""Pure action -> Ackermann mapping (no ROS imports, easily unit-tested).

Mirrors ``f1tenth_env/env.py`` ``_apply_actions``: throttle scales speed by
``max_speed`` and steering scales the center steering angle by ``max_steer``.
Negative throttle is a brake; ``brake_behavior`` controls whether that means a hard
stop (speed 0) or reverse (negative speed).
"""

from __future__ import annotations


def map_action_to_drive(
    throttle: float,
    steering: float,
    max_speed: float,
    max_steer: float,
    clip_actions: float = 1.0,
    brake_behavior: str = "stop",
) -> tuple[float, float]:
    """Return ``(speed_mps, steering_angle_rad)`` for an AckermannDrive command."""
    throttle = max(-clip_actions, min(clip_actions, float(throttle)))
    steering = max(-clip_actions, min(clip_actions, float(steering)))

    if brake_behavior == "reverse":
        speed = throttle * max_speed
    else:  # "stop": negative throttle commands a stop, not reverse
        speed = max(throttle, 0.0) * max_speed

    steering_angle = steering * max_steer
    return speed, steering_angle
