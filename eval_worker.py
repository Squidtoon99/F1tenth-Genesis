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
        if extra["metrics"]["lap_count"] > current_lap:
            if current_lap_time >= 10.0:  # filter out invalid lap times
                lap_times.append(current_lap_time)
                
            current_lap_time = 0.0
            current_lap += 1


    lap_times = [lap_time for lap_time in lap_times if lap_time >= 10.0]
    data["lap_times"] = lap_times

    # Tracking progress
    progress = [extra["metrics"]["progress"] for extra in extras]
    data["progress"] = progress

    
    return data
