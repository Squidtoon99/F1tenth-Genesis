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
    delta_center: torch.Tensor, L: float, W: float
) -> torch.Tensor:
    """
    Convert desired 'center' steering angles to left/right hinge angles.
    
    Args:
        delta_center: (N,) tensor of center steering angles in radians
        L: wheelbase (meters)
        W: track width (meters)
    
    Returns:
        (N, 2) tensor of [left, right] steering angles
    """
    # Near-zero steering: both wheels straight
    small_angle_mask = torch.abs(delta_center) < 1e-6
    
    # Turning radius of the vehicle centerline: R = L / tan(delta)
    R = L / torch.tan(delta_center)
    
    # Left/right wheel radii
    R_left = R - (W / 2.0)
    R_right = R + (W / 2.0)
    
    # Wheel steering angles
    delta_left = torch.atan(L / R_left)
    delta_right = torch.atan(L / R_right)
    
    # Apply zero steering for near-zero angles
    delta_left = torch.where(small_angle_mask, torch.zeros_like(delta_left), delta_left)
    delta_right = torch.where(small_angle_mask, torch.zeros_like(delta_right), delta_right)
    
    return torch.stack([delta_left, delta_right], dim=1)


def wheel_omegas_from_v(
    delta_center: torch.Tensor, v: torch.Tensor, r: float, L: float, W: float
) -> torch.Tensor:
    """
    Compute per-wheel angular velocities (rad/s) for a simple Ackermann model.
    
    Args:
        delta_center: (N,) tensor of center steering angles in radians
        v: (N,) tensor of vehicle speeds (m/s)
        r: wheel radius (meters)
        L: wheelbase (meters)
        W: track width (meters)
    
    Returns:
        (N, 4) tensor of wheel angular velocities [LR, RR, LF, RF]
    """
    # Straight-line case: all wheels same speed
    small_angle_mask = torch.abs(delta_center) < 1e-6
    omega_straight = v / r
    
    # Turning case: R = L / tan(delta)
    R = L / torch.tan(delta_center)
    
    # Approximate left/right ground speeds based on turn radius
    v_left = v * (R - W / 2.0) / R
    v_right = v * (R + W / 2.0) / R
    
    # Convert to angular velocities
    omega_left = v_left / r
    omega_right = v_right / r
    
    # Apply straight-line values for near-zero angles
    omega_left = torch.where(small_angle_mask, omega_straight, omega_left)
    omega_right = torch.where(small_angle_mask, omega_straight, omega_right)
    
    # Rear and front on each side share the same speed: [LR, RR, LF, RF]
    return torch.stack([omega_left, omega_right, omega_left, omega_right], dim=1)


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
