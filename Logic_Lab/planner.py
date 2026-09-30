"""Simple logical planner: states are sets of propositions, BFS over applicable actions."""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Action:
    name: str
    pos_pre: frozenset = frozenset()   # facts that must be true
    neg_pre: frozenset = frozenset()   # facts that must be false
    pos_eff: frozenset = frozenset()   # facts added
    neg_eff: frozenset = frozenset()   # facts removed


def applicable(state, action):
    """S |= Preconditions(a): all positive preconditions hold, no negative one does."""
    return action.pos_pre <= state and not (action.neg_pre & state)


def apply_action(state, action):
    """Remove negative effects first, then add positive effects."""
    return frozenset((state - action.neg_eff) | action.pos_eff)


def bfs_plan(initial, actions, goal, check_preconditions=True):
    """Breadth-first search. Returns [(action, state_after), ...] or None if no plan exists.
    check_preconditions=False exists only to demonstrate the bug it causes (see tests)."""
    initial, goal = frozenset(initial), frozenset(goal)
    if goal <= initial:
        return []
    frontier = deque([(initial, [])])
    visited = {initial}
    while frontier:
        state, path = frontier.popleft()
        for a in actions:
            if check_preconditions and not applicable(state, a):
                continue
            nxt = apply_action(state, a)
            if nxt in visited:
                continue
            new_path = path + [(a, nxt)]
            if goal <= nxt:                       # goal test: G is a subset of the state
                return new_path
            visited.add(nxt)
            frontier.append((nxt, new_path))
    return None


def validate_plan(initial, plan_actions, goal):
    """Independent checker: replays a list of Actions from the initial state, checking every
    precondition and the goal. Returns (ok, list_of_log_lines)."""
    state, log = frozenset(initial), []
    for i, a in enumerate(plan_actions, 1):
        missing = sorted(a.pos_pre - state)
        blocked = sorted(a.neg_pre & state)
        if missing or blocked:
            log.append(f"step {i}: {a.name} NOT applicable (missing {missing}, forbidden present {blocked})")
            return False, log
        state = apply_action(state, a)
        log.append(f"step {i}: {a.name} OK -> {sorted(state)}")
    ok = frozenset(goal) <= state
    log.append(f"goal {sorted(goal)} {'satisfied' if ok else 'NOT satisfied'}")
    return ok, log


# ---------------------------------------------------------------- warehouse problem
LOCS = ["A", "B", "C"]
EDGES = [("A", "B"), ("B", "A"), ("B", "C"), ("C", "B")]


def F(*facts):
    return frozenset(facts)


def warehouse_actions(include_pickup=True):
    acts = []
    for x, y in EDGES:
        acts.append(Action(f"Move({x},{y})", pos_pre=F(f"At(Robot,{x})"),
                           pos_eff=F(f"At(Robot,{y})"), neg_eff=F(f"At(Robot,{x})")))
    for l in LOCS:
        if include_pickup:
            acts.append(Action(f"PickUp(Package,{l})", pos_pre=F(f"At(Robot,{l})", f"At(Package,{l})"),
                               pos_eff=F("Holding(Package)"), neg_eff=F(f"At(Package,{l})")))
        acts.append(Action(f"Drop(Package,{l})", pos_pre=F(f"At(Robot,{l})", "Holding(Package)"),
                           pos_eff=F(f"At(Package,{l})"), neg_eff=F("Holding(Package)")))
    return acts


INITIAL = F("At(Robot,A)", "At(Package,A)")
GOAL = F("At(Package,C)")


def show(initial, plan):
    if plan is None:
        print("No plan found")
        return
    print("S0:", sorted(initial))
    for i, (a, s) in enumerate(plan, 1):
        print(f"a{i} = {a.name:22s} S{i}: {sorted(s)}")
    if not plan:
        print("(goal already satisfied, empty plan)")


if __name__ == "__main__":
    print("=== warehouse problem ===")
    show(INITIAL, bfs_plan(INITIAL, warehouse_actions(), GOAL))
