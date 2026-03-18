import math

import torch


def validate_observation_ranges(obs_buf: torch.Tensor) -> None:
    if not torch.isfinite(obs_buf).all():
        raise ValueError("Observation buffer contains NaN or Inf values")

    if obs_buf.shape[1] < 10:
        raise ValueError(f"Observation dimension too small: {obs_buf.shape[1]}")

    track_progress = obs_buf[:, 6:8]
    if (track_progress.abs() > 1.0001).any():
        raise ValueError("track_progress term out of expected [-1, 1] range")

    centerline_angle = obs_buf[:, 8]
    if (
        (centerline_angle < -math.pi - 1e-4) | (centerline_angle > math.pi + 1e-4)
    ).any():
        raise ValueError("centerline_angle term out of expected [-pi, pi] range")

    contact_flag = obs_buf[:, 9]
    invalid_contact = (contact_flag != 0.0) & (contact_flag != 1.0)
    if invalid_contact.any():
        raise ValueError("contact_flag term must be binary (0/1)")


def validate_reward_ranges(
    reward_buf: torch.Tensor,
    last_reward_terms: dict[str, torch.Tensor] | None,
) -> None:
    if not torch.isfinite(reward_buf).all():
        raise ValueError("Reward buffer contains NaN or Inf values")

    if not last_reward_terms:
        return

    oob_term = last_reward_terms.get("oob_penalty")
    if oob_term is not None and (oob_term > 1e-5).any():
        raise ValueError("oob_penalty became positive; expected <= 0")

    progress_term = last_reward_terms.get("progress")
    if progress_term is not None and not torch.isfinite(progress_term).all():
        raise ValueError("progress reward term contains NaN or Inf")
