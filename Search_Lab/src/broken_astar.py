"""broken_astar.py - deliberately WRONG versions of A* (Lab Task 3 / 7).

The tests must catch every one of these. Each bug is one small change to the real
algorithm, the kind of mistake a generated program could plausibly contain.

  no_closed_set : never checks for repeated states (re-expands the same cell again and again)
  ignore_g      : priority = h only (this is greedy best-first, not A*)
  overestimate  : uses 3 * Manhattan as the heuristic (not admissible)
  through_walls : forgets to reject '#' cells (only checks the map edge)
  off_by_one    : reports the number of cells on the path instead of the number of moves
"""
import heapq
from warehouse import SearchResult, neighbours, h_manhattan, ACTIONS

BUGS = ["no_closed_set", "ignore_g", "overestimate", "through_walls", "off_by_one"]


def _neighbours_no_wall_check(grid, state):
    """Buggy transition: only checks the map boundary, walls are ignored."""
    r, c = state
    for name, (dr, dc) in ACTIONS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
            yield name, (nr, nc)


def astar_buggy(grid, start, goal, bug, max_expansions=3000):
    """A* with one injected bug. max_expansions stops the run that would never end.

    Each frontier entry carries its own parent chain (node = (state, parent_node)),
    because without a closed set a shared parent dict can contain cycles."""
    h = (lambda s, g: 3 * h_manhattan(s, g)) if bug == "overestimate" else h_manhattan
    nbrs = _neighbours_no_wall_check if bug == "through_walls" else neighbours
    counter = 0
    frontier = [(h(start, goal), counter, 0, (start, None))]
    best_g = {start: 0}
    expanded_states = set()
    expanded = 0
    generated = 1
    while frontier:
        _f, _c, g, node = heapq.heappop(frontier)
        state = node[0]
        if bug != "no_closed_set" and g > best_g[state]:
            continue
        expanded += 1
        expanded_states.add(state)
        if expanded > max_expansions:
            break
        if state == goal:
            path = []
            while node is not None:
                path.append(node[0])
                node = node[1]
            path.reverse()
            cost = len(path) if bug == "off_by_one" else len(path) - 1
            return SearchResult(True, path, cost, expanded, len(expanded_states), generated)
        for _a, nxt in nbrs(grid, state):
            new_g = g + 1
            if bug == "no_closed_set" or nxt not in best_g or new_g < best_g[nxt]:
                best_g[nxt] = new_g
                counter += 1
                priority = h(nxt, goal) if bug == "ignore_g" else new_g + h(nxt, goal)
                heapq.heappush(frontier, (priority, counter, new_g, (nxt, node)))
                generated += 1
    return SearchResult(False, None, None, expanded, len(expanded_states), generated)
