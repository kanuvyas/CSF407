"""Tests A, B, C from the lab plus a few extra checks. Run:  python3 -m pytest -q tests"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from planner import *

ACTS = {a.name: a for a in warehouse_actions()}
names = lambda plan: [a.name for a, _ in plan]


def test_A_solvable_problem():
    plan = bfs_plan(INITIAL, warehouse_actions(), GOAL)
    assert names(plan) == ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]
    ok, _ = validate_plan(INITIAL, [a for a, _ in plan], GOAL)
    assert ok and GOAL <= plan[-1][1]


def test_B_impossible_when_pickup_removed():
    assert bfs_plan(INITIAL, warehouse_actions(include_pickup=False), GOAL) is None


def test_C_robot_at_C_is_not_package_at_C():
    # goal only about the robot: the plan just moves, the package stays at A
    plan = bfs_plan(INITIAL, warehouse_actions(), F("At(Robot,C)"))
    assert names(plan) == ["Move(A,B)", "Move(B,C)"]
    final = plan[-1][1]
    assert "At(Package,A)" in final and "At(Package,C)" not in final
    # and without PickUp, reaching C with the robot does not give the package goal
    assert bfs_plan(INITIAL, warehouse_actions(include_pickup=False), GOAL) is None


def test_task0_applicability():
    assert applicable(INITIAL, ACTS["PickUp(Package,A)"])
    assert not applicable(INITIAL, ACTS["Drop(Package,C)"])
    assert not applicable(INITIAL, ACTS["PickUp(Package,B)"])


def test_example_plan_from_lab_handout_is_invalid():
    bad = [ACTS[n] for n in ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)", "Drop(Package,C)"]]
    ok, log = validate_plan(INITIAL, bad, GOAL)
    assert not ok and "PickUp(Package,B) NOT applicable" in log[1]


def test_planner_without_precondition_check_invents_a_plan():
    plan = bfs_plan(INITIAL, warehouse_actions(), GOAL, check_preconditions=False)
    assert names(plan) == ["Drop(Package,C)"]             # "delivers" the package from A instantly
    ok, _ = validate_plan(INITIAL, [a for a, _ in plan], GOAL)
    assert not ok                                        # the independent validator rejects it
