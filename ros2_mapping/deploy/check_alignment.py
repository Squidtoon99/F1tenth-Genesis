#!/usr/bin/env python3
import argparse, subprocess, re, sys
import numpy as np

def ros_echo(topic, timeout=5):
    r = subprocess.run(
        ["timeout", str(timeout), "ros2", "topic", "echo", topic, "--no-arr"],
        capture_output=True, text=True,
    )
    return r.stdout

def parse_obs(text):
    vals = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- "):
            try:
                vals.append(float(s[2:]))
            except ValueError:
                continue
    return vals

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="/home/shereef/maps/f1tenth_map_centerline.csv")
    args = p.parse_args()
    pose_txt = ros_echo("/pf/pose/odom")
    mx = re.search(r"x: ([-\d.e+]+)", pose_txt)
    my = re.search(r"y: ([-\d.e+]+)", pose_txt)
    if not mx:
        print("No /pf/pose/odom")
        return 1
    px, py = float(mx.group(1)), float(my.group(1))
    data = np.genfromtxt(args.csv, delimiter=",", names=True, comments="#")
    cl = np.stack([data["x_m"], data["y_m"]], axis=1)
    d = np.linalg.norm(cl - (px, py), axis=1)
    i = int(np.argmin(d))
    print(f"PF pose=({px:.2f},{py:.2f}) nearest centerline dist={d[i]:.2f}m idx={i}")
    obs = parse_obs(ros_echo("/rl/observation", 6))
    if len(obs) >= 387:
        print(f"obs dim={len(obs)} ey={obs[10]:.3f} contact={obs[11]:.0f} opp_present={obs[386]:.0f}")
        if abs(obs[10]) > 1.0:
            print("WARN: |ey|>1m — refine pose or move car onto centerline")
        else:
            print("PASS: lateral error within 1m (refine to <0.3m before live)")
    else:
        print(f"obs count={len(obs)} (expect 387)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
