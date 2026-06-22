#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"
CSV="${CSV:-$HOME/maps/f1tenth_map_centerline.csv}"
python3 "$HOME/deploy/check_alignment.py" --csv "$CSV"
