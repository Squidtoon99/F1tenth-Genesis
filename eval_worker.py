from config import Config
from typing import Any


def process_eval_data(cfg: "Config", task: Any, extras: Any, traj: list) -> dict:
    data = {}
    # Lap timing
    lap_times = []
    current_lap_time = 0.0
    current_lap = 0
    for extra, step in zip(extras, traj):
        current_lap_time += 0.1  # sim dt (hardcoded)
        if extra.get("lap_complete"):
            lap_times.append(current_lap_time)
            current_lap_time = 0.0
            current_lap += 1

    data["lap_times"] = lap_times
    return data
