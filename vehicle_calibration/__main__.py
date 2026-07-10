"""Unified CLI for the vehicle calibration toolkit.

    python -m vehicle_calibration profile genesis --run-id <id> [--overrides JSON]
    python -m vehicle_calibration profile irl     --run-id <id>   # prints how-to
    python -m vehicle_calibration parse bag <bag_path> --run-id <id>
    python -m vehicle_calibration compare         --run-id <id>
    python -m vehicle_calibration fit             --run-id <id>
    python -m vehicle_calibration lap-compare <subcmd> ...
    python -m vehicle_calibration track-align <subcmd> ...

Heavy imports (Genesis, ROS) are deferred into each subcommand so unrelated
commands stay fast and importable without those dependencies installed.
"""

from __future__ import annotations

import sys
import textwrap

from . import run_dir

_USAGE = textwrap.dedent(__doc__)


def _profile(argv: list[str]) -> int:
    if not argv or argv[0] not in ("genesis", "irl"):
        print("usage: profile {genesis|irl} ...")
        return 2
    which, rest = argv[0], argv[1:]
    if which == "genesis":
        from .profile_genesis import main as gmain

        return gmain(rest)
    return _profile_irl(rest)


def _profile_irl(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="profile irl")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-delay-s", default="3.0")
    args = parser.parse_args(argv)
    bag = run_dir(args.run_id) + "/bag"
    print(
        textwrap.dedent(
            f"""\
            IRL open-loop profiling is an on-car procedure (no checkpoint needed):

            1. Bring up the lean stack WITHOUT policy_inference (the profiler owns
               /rl/action). The drive node applies the staged speed_limit_mps cap:
                 ros2 launch f1tenth_rl_vehicle bringup_vehicle.launch.py \\
                     enable_profiler:=true
               (or run profile_maneuvers.launch.py alongside your usual bringup)

            2. Arm teleop override, then start recording into this run's bag dir:
                 ros2 bag record -o {bag} \\
                     /rl/action /drive /odom /pf/pose/odom /rl/observation \\
                     /calib/maneuver /calib/role

            3. After the schedule completes (node latches a stop), Ctrl-C the bag,
               then parse it back on a machine with ROS:
                 python -m vehicle_calibration parse bag {bag} --run-id {args.run_id}

            Raise speed_limit_mps to 5-6 m/s before the fit-band maneuvers so the
            >= 3 m/s windows actually reach race speed.
            """
        )
    )
    return 0


def _parse(argv: list[str]) -> int:
    if not argv or argv[0] != "bag":
        print("usage: parse bag <bag_path> --run-id <id>")
        return 2
    from .parse_bag import main as pmain

    return pmain(argv[1:])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(_USAGE)
        return 0

    cmd, rest = argv[0], argv[1:]
    if cmd == "profile":
        return _profile(rest)
    if cmd == "parse":
        return _parse(rest)
    if cmd == "compare":
        from .compare import main as cmain

        return cmain(rest)
    if cmd == "fit":
        from .fit import main as fmain

        return fmain(rest)
    if cmd == "lap-compare":
        from .lap_compare import main as lmain

        return lmain(rest)
    if cmd == "track-align":
        from .track_align import main as tmain

        return tmain(rest)

    print(f"unknown command: {cmd}\n")
    print(_USAGE)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
