#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"
CSV="${CSV:-$HOME/maps/f1tenth_map_centerline.csv}"
IDX="${IDX:-0}"
python3 "${DEPLOY:-$HOME/deploy}/set_initial_pose.py" --csv "$CSV" --index "$IDX" --repeat 5
echo "Initial pose set from centerline index $IDX. Expect obs[10] ~ 0 when car is on track."
