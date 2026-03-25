"""
Convert car values
"""

import os

import numpy as np
import torch
import genesis as gs

URDF_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "F110.export.urdf")
)

# Your joints
WHEEL_JOINTS = [
    "left_rear_wheel_joint",
    "right_rear_wheel_joint",
    "left_front_wheel_joint",
    "right_front_wheel_joint",
]

STEER_JOINTS = [
    "left_steering_hinge_joint",
    "right_steering_hinge_joint",
]

# Parsed from your URDF
WHEEL_RADIUS = 0.05  # meters
WHEELBASE = 0.3302  # meters (front axle x - rear axle x)
TRACK_WIDTH = 0.20  # meters (left y - right y)

# Gains computed from URDF limits
KP_WHEELS = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
KV_WHEELS = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)

KP_STEER = np.array([8.0, 8.0], dtype=np.float32)
KV_STEER = np.array([0.8, 0.8], dtype=np.float32)

# Effort limits from URDF
EFF_WHEELS = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)
EFF_STEER = np.array([10.0, 10.0], dtype=np.float32)

CHASSIS_FRICTION = 0.05
WHEEL_FRICTION = 1.0


def ackermann_left_right(
    delta_center: float, L: float, W: float
) -> tuple[float, float]:
    """
    Convert a desired 'center' steering angle to left/right hinge angles.
    delta_center in radians. Clamps internally for numerical stability.
    """
    # Near-zero steering: both wheels straight
    if abs(delta_center) < 1e-6:
        return 0.0, 0.0

    # Turning radius of the vehicle centerline
    # R = L / tan(delta)
    R = L / np.tan(delta_center)

    # Left/right wheel radii
    R_left = R - (W / 2.0)
    R_right = R + (W / 2.0)

    # Wheel steering angles
    delta_left = np.arctan(L / R_left)
    delta_right = np.arctan(L / R_right)
    return float(delta_left), float(delta_right)


def wheel_omegas_from_v(
    delta_center: float, v: float, r: float, L: float, W: float
) -> np.ndarray:
    """
    Compute per-wheel angular velocities (rad/s) for a simple Ackermann model.
    Order: [LR, RR, LF, RF]
    """
    if abs(delta_center) < 1e-6:
        omega = v / r
        return np.array([omega, omega, omega, omega], dtype=np.float32)

    R = L / np.tan(delta_center)

    # Approximate left/right ground speeds based on turn radius
    v_left = v * (R - W / 2.0) / R
    v_right = v * (R + W / 2.0) / R

    # Rear and front on each side share the same speed in this simple model
    return np.array(
        [v_left / r, v_right / r, v_left / r, v_right / r], dtype=np.float32
    )


def setup_entity_controls(car) -> tuple[list[int], list[int]]:
    # Map joint names -> local dof indices (Genesis convention)
    wheel_dofs = [car.get_joint(n).dof_idx_local for n in WHEEL_JOINTS]
    steer_dofs = [car.get_joint(n).dof_idx_local for n in STEER_JOINTS]

    # Set gains and force limits :contentReference[oaicite:3]{index=3}
    car.set_dofs_kp(KP_WHEELS, wheel_dofs)
    car.set_dofs_kv(KV_WHEELS, wheel_dofs)
    car.set_dofs_force_range(-EFF_WHEELS, EFF_WHEELS, wheel_dofs)

    car.set_dofs_kp(KP_STEER, steer_dofs)
    car.set_dofs_kv(KV_STEER, steer_dofs)
    car.set_dofs_force_range(-EFF_STEER, EFF_STEER, steer_dofs)

    # join passive damping
    car.set_dofs_damping(
        np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32), wheel_dofs
    )
    car.set_dofs_frictionloss(
        np.array([0.02, 0.02, 0.02, 0.02], dtype=np.float32), wheel_dofs
    )

    car.set_friction(CHASSIS_FRICTION)

    link_names = [
        "base_link",
        "left_rear_wheel",
        "right_rear_wheel",
        "left_front_wheel",
        "right_front_wheel",
    ]
    link_ids = [car.get_link(name).idx_local for name in link_names]

    # baseline friction is CHASSIS_FRICTION
    # ratio = desired / baseline
    ratios = np.array(
        [
            1.0,  # chassis -> CHASSIS_FRICTION
            WHEEL_FRICTION / CHASSIS_FRICTION,
            WHEEL_FRICTION / CHASSIS_FRICTION,
            WHEEL_FRICTION / CHASSIS_FRICTION,
            WHEEL_FRICTION / CHASSIS_FRICTION,
        ],
        dtype=np.float32,
    )

    ratios_t = torch.from_numpy(ratios[None, :]).to(device=gs.device)
    car.set_friction_ratio(ratios_t, links_idx_local=link_ids)
    return (wheel_dofs, steer_dofs)


def wheel_torques_from_throttle(
    throttle: np.ndarray, max_torque: float = 2.5
) -> np.ndarray:
    """
    Map normalized throttle in [-1, 1] to wheel torques.
    Start conservative; raise slowly if needed.
    """
    throttle = np.clip(throttle, -1.0, 1.0).astype(np.float32)
    tau = throttle[:, None] * max_torque
    return np.repeat(tau, 4, axis=1).astype(np.float32)
