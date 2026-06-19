"""Pure tests for the action -> Ackermann mapping."""

import math

from f1tenth_rl_agent import interfaces as ifc
from f1tenth_rl_agent.drive_math import map_action_to_drive


def test_full_throttle_zero_steer():
    speed, steer = map_action_to_drive(1.0, 0.0, ifc.MAX_SPEED, ifc.MAX_STEER)
    assert math.isclose(speed, ifc.MAX_SPEED)
    assert math.isclose(steer, 0.0)


def test_brake_stops_by_default():
    speed, steer = map_action_to_drive(-1.0, 0.0, ifc.MAX_SPEED, ifc.MAX_STEER)
    assert math.isclose(speed, 0.0)


def test_reverse_behavior():
    speed, _ = map_action_to_drive(
        -1.0, 0.0, ifc.MAX_SPEED, ifc.MAX_STEER, brake_behavior="reverse"
    )
    assert math.isclose(speed, -ifc.MAX_SPEED)


def test_full_left_steer():
    _, steer = map_action_to_drive(0.0, 1.0, ifc.MAX_SPEED, ifc.MAX_STEER)
    assert math.isclose(steer, ifc.MAX_STEER)


def test_clipping():
    speed, steer = map_action_to_drive(5.0, -5.0, ifc.MAX_SPEED, ifc.MAX_STEER)
    assert math.isclose(speed, ifc.MAX_SPEED)
    assert math.isclose(steer, -ifc.MAX_STEER)
