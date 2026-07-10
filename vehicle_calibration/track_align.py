"""Phase 0 helpers: get the real-track centerline into the localization frame.

Policy-lap validation is only meaningful once the surveyed/SLAM centerline is
aligned to the particle-filter map frame. This module does not invent a mapping
pipeline; it wraps the existing tooling and automates the mechanical bits:

  * ``check``    - print the on-car obs[10] alignment gate + bag record command.
  * ``validate`` - run ros2_mapping's validate_track_alignment against a map.
  * ``bundle``   - copy a verified centerline CSV into the two deploy asset dirs
                   (the same locations scripts/resample_centerline.py writes to).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BUNDLED_DIRS = (
    os.path.join(ROOT, "ros2_deploy", "f1tenth_rl_agent", "assets"),
    os.path.join(ROOT, "ros2_deploy", "assets"),
)

_CHECK_TEXT = """\
Phase 0 - real-track centerline alignment gate
==============================================
1. Place the car at a known surveyed pose on the carpet (e.g. start/finish line).
2. Bring up localization + the lean stack (no throttle yet):
     ros2 launch f1tenth_rl_vehicle bringup_vehicle.launch.py \\
         track_csv:=/abs/path/<TRACK>_centerline.csv
3. Confirm the signed lateral error in the observation is ~0 at the known pose:
     ros2 topic echo /rl/observation --field data | sed -n '11p'   # index 10
   Repeat at >= 2 distinct known poses; |obs[10]| should be within a few cm.
4. Quantify map quality offline:
     python -m vehicle_calibration track-align validate \\
         --map /abs/map.yaml --reference /abs/<TRACK>_centerline.csv
   Relaxed gate: centerline mean < 1.0 m, p95 < 2.0 m, drivable fraction > 0.8.
5. If SLAM jitter/seam is visible, smooth first:
     python scripts/resample_centerline.py --track <TRACK>          # preview
     python scripts/resample_centerline.py --track <TRACK> --deploy # write
6. Bundle the verified CSV into the deploy assets:
     python -m vehicle_calibration track-align bundle --csv /abs/<TRACK>_centerline.csv
7. Record a shakedown/lap bag for later analysis:
     ros2 bag record -o vehicle_calibration/runs/<RUN_ID>/bag \\
         /rl/observation /rl/action /drive /pf/pose/odom /odom

Only after this gate passes is the policy-lap comparison (lap_compare) meaningful.
"""


def cmd_check(_args) -> int:
    print(_CHECK_TEXT)
    return 0


def cmd_validate(args) -> int:
    script = os.path.join(
        ROOT, "ros2_mapping", "postprocess", "validate_track_alignment.py"
    )
    if not os.path.isfile(script):
        print(f"[track-align] not found: {script}")
        return 1
    cmd = [sys.executable, script, "--map", args.map, "--reference", args.reference]
    if args.strict:
        cmd.append("--strict")
    print("[track-align] running:", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_bundle(args) -> int:
    csv_path = args.csv
    if not os.path.isfile(csv_path):
        print(f"[track-align] not found: {csv_path}")
        return 1
    name = os.path.basename(csv_path)
    if not name.endswith("_centerline.csv"):
        print("[track-align] warning: expected a *_centerline.csv filename")
    for d in BUNDLED_DIRS:
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, name)
        shutil.copyfile(csv_path, dst)
        print(f"[track-align] bundled -> {dst}")
    print(
        "[track-align] set vehicle.yaml track_csv (or launch track_csv:=) to this "
        f"asset name: {name}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 0 track-alignment helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="print the on-car alignment gate checklist")

    pv = sub.add_parser("validate", help="run ros2_mapping alignment validation")
    pv.add_argument("--map", required=True, help="occupancy map .yaml")
    pv.add_argument("--reference", required=True, help="reference centerline CSV")
    pv.add_argument("--strict", action="store_true")

    pb = sub.add_parser("bundle", help="copy a verified centerline into deploy assets")
    pb.add_argument("--csv", required=True)

    args = parser.parse_args(argv)
    return {"check": cmd_check, "validate": cmd_validate, "bundle": cmd_bundle}[
        args.cmd
    ](args)


if __name__ == "__main__":
    raise SystemExit(main())
