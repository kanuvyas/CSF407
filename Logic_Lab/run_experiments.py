"""Prints the results recorded in REPORT.md (Tasks 0, 1, 3, 5)."""
from planner import *

acts = warehouse_actions(); A = {a.name: a for a in acts}
print("##### Task 0: applicability in the initial state", sorted(INITIAL))
for n in ["PickUp(Package,A)", "Drop(Package,C)"]:
    print(f"{n:20s} applicable: {applicable(INITIAL, A[n])}  (missing: {sorted(A[n].pos_pre - INITIAL)})")

print("\n##### Task 1: manual plan, replayed by the independent validator")
manual = [A[n] for n in ["PickUp(Package,A)", "Move(A,B)", "Move(B,C)", "Drop(Package,C)"]]
print("\n".join(validate_plan(INITIAL, manual, GOAL)[1]))

print("\n##### Test A: original problem, goal", sorted(GOAL))
plan = bfs_plan(INITIAL, acts, GOAL); show(INITIAL, plan)
print("\n".join(validate_plan(INITIAL, [a for a, _ in plan], GOAL)[1]))

print("\n##### Test B: PickUp removed, goal", sorted(GOAL))
show(INITIAL, bfs_plan(INITIAL, warehouse_actions(include_pickup=False), GOAL))

print("\n##### Test C1: goal At(Robot,C) only")
g = F("At(Robot,C)"); plan = bfs_plan(INITIAL, acts, g); show(INITIAL, plan)
print("package still at A:", "At(Package,A)" in plan[-1][1], "| package at C:", "At(Package,C)" in plan[-1][1])
print("\n##### Test C2: PickUp removed, goal At(Package,C) (robot can still reach C)")
show(INITIAL, bfs_plan(INITIAL, warehouse_actions(include_pickup=False), GOAL))

print("\n##### Extra: example sequence from the lab handout (Move(A,B), PickUp(Package,B), ...)")
bad = [A[n] for n in ["Move(A,B)", "PickUp(Package,B)", "Move(B,C)", "Drop(Package,C)"]]
print("\n".join(validate_plan(INITIAL, bad, GOAL)[1]))

print("\n##### Extra: planner with the precondition check switched off (bug demo)")
plan = bfs_plan(INITIAL, acts, GOAL, check_preconditions=False); show(INITIAL, plan)
print("\n".join(validate_plan(INITIAL, [a for a, _ in plan], GOAL)[1]))
