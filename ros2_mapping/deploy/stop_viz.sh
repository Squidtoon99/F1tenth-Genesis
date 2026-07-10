#!/usr/bin/env bash
# Stop RViz / noVNC desktop stack (use Foxglove instead).
pkill -f rviz2 2>/dev/null || true
pkill -f websockify 2>/dev/null || true
pkill -f x11vnc 2>/dev/null || true
pkill -f "Xvfb :" 2>/dev/null || true
rm -f /tmp/novnc_rviz/*.pid 2>/dev/null || true
echo "Stopped rviz2 / noVNC / Xvfb"
