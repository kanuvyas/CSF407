"""
broken_agents.py - Deliberately wrong planners used to check that my tests work.

Lab: Task 3 (testing and validating LLM-generated code).
Each function takes a map (list of strings) and returns a list of action
names, or None for "no path". Each one has ONE bug, described in its docstring.
tests/test_agent.py must reject every one of them. Do not use these for real.
"""

from collections import deque

from warehouse_agent import ACTIONS, parse_map

CAP = 20000  # give up after this many expansions (used by the no-visited-set bug)


def _bfs(map_lines, *, check_walls=True, use_visited=True, swap_up_down=False,
         none_becomes_empty=False):
    """A copy of BFS with switches that switch individual bugs on."""
    grid, start, goal = parse_map(map_lines)
    rows, cols = len(grid), len(grid[0])
    names = {"Up": "Down", "Down": "Up"} if swap_up_down else {}
    frontier = deque([(start, [])])
    seen = {start}
    expanded = 0
    while frontier:
        state, actions = frontier.popleft()
        expanded += 1
        if expanded > CAP:
            return None
        if state == goal:
            return actions
        for name, (dr, dc) in ACTIONS:
            r, c = state[0] + dr, state[1] + dc
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if check_walls and grid[r][c] == "#":
                continue
            if use_visited and (r, c) in seen:
                continue
            seen.add((r, c))
            frontier.append(((r, c), actions + [names.get(name, name)]))
    return [] if none_becomes_empty else None


def walks_through_walls(map_lines):
    """BUG: obstacles are ignored, so the path can cross '#' squares."""
    return _bfs(map_lines, check_walls=False)


def no_visited_set(map_lines):
    """BUG: no visited set, so the search revisits squares and explodes."""
    return _bfs(map_lines, use_visited=False)


def swapped_up_down(map_lines):
    """BUG: the Up and Down labels are swapped in the returned plan."""
    return _bfs(map_lines, swap_up_down=True)


def empty_plan_when_unreachable(map_lines):
    """BUG: returns [] (looks like 'already there') when the goal is unreachable."""
    return _bfs(map_lines, none_becomes_empty=True)


def depth_first(map_lines):
    """BUG (of choice): DFS finds A path, but not the shortest one."""
    grid, start, goal = parse_map(map_lines)
    rows, cols = len(grid), len(grid[0])
    stack = [(start, [])]
    seen = set()
    while stack:
        state, actions = stack.pop()          # LIFO stack instead of FIFO queue
        if state == goal:
            return actions
        if state in seen:
            continue
        seen.add(state)
        for name, (dr, dc) in ACTIONS:
            r, c = state[0] + dr, state[1] + dc
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] != "#":
                stack.append(((r, c), actions + [name]))
    return None


BROKEN = {
    "walks_through_walls": walks_through_walls,
    "no_visited_set": no_visited_set,
    "swapped_up_down": swapped_up_down,
    "empty_plan_when_unreachable": empty_plan_when_unreachable,
    "depth_first": depth_first,
}
