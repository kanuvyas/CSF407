"""astar.py - A* search on the warehouse grid (Lab Task 2, code inspected in Task 4).

Design decisions (mine, written down before generating the code):
  * frontier      : binary heap (heapq) ordered by (f, h, insertion counter)
  * tie-breaking  : lowest f first, then lowest h (closer to goal), then oldest
  * goal test     : done when the goal is REMOVED from the frontier
  * repeated states: best_g remembers the cheapest known g per state; a state is
                     only pushed again if a strictly cheaper path is found.
                     Stale heap entries are skipped when popped.
"""
import heapq
from warehouse import SearchResult, neighbours, h_manhattan


def astar(grid, start, goal, h=h_manhattan, max_expansions=None):
    """Run A* from start to goal using heuristic h(state, goal).

    Returns a SearchResult. Never loops forever: every state can only be pushed
    again with a strictly smaller g, and g values are non-negative integers."""
    counter = 0                                   # insertion order, used as last tie-break
    h0 = h(start, goal)
    frontier = [(0 + h0, h0, counter, 0, start)]  # entries: (f, h, counter, g, state)
    best_g = {start: 0}                           # cheapest g found so far per state (also "visited")
    parent = {start: None}                        # for path reconstruction
    expanded_states = set()                       # distinct states expanded
    expanded = 0
    generated = 1

    while frontier:
        f, hn, _, g, state = heapq.heappop(frontier)   # select lowest f = g + h
        if g > best_g[state]:
            continue                                   # stale entry: a cheaper path was found later
        expanded += 1                                  # COUNT: this state is expanded now
        expanded_states.add(state)
        if max_expansions is not None and expanded > max_expansions:
            break                                      # safety cap (only used in tests)

        if state == goal:                              # goal test on removal
            path = []
            while state is not None:                   # walk parents back to S
                path.append(state)
                state = parent[state]
            path.reverse()
            return SearchResult(True, path, len(path) - 1, expanded,
                                len(expanded_states), generated)

        for _action, nxt in neighbours(grid, state):   # apply every valid action
            new_g = g + 1                              # every move costs 1
            if nxt not in best_g or new_g < best_g[nxt]:   # new state, or cheaper route
                best_g[nxt] = new_g
                parent[nxt] = state
                counter += 1
                hn2 = h(nxt, goal)
                heapq.heappush(frontier, (new_g + hn2, hn2, counter, new_g, nxt))  # f = g + h
                generated += 1

    return SearchResult(False, None, None, expanded, len(expanded_states), generated)
