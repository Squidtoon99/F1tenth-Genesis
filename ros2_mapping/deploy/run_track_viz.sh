#!/usr/bin/env bash
# Publish centerline + left/right boundaries in RViz (requires map_server + Fixed Frame=map).
source /opt/ros/humble/setup.bash
source "$HOME/f1tenth_ws/install/setup.bash"

TRACK="${TRACK:-$HOME/maps/f1tenth_map_centerline.csv}"
ros2 launch f1tenth_mapping track_viz.launch.py track_csv:="$TRACK" frame_id:=map
