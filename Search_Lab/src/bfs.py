"""bfs.py - breadth-first search on the same warehouse problem (Lab Task 5).

Uses the same conventions as astar.py so the counts are comparable: the goal test
happens when a state is removed from the frontier, and that removal is counted as
an expansion."""
from collections import deque
from warehouse import SearchResult, neighbours


def bfs(grid, start, goal):
    """Breadth-first search; returns a SearchResult (shortest path, all costs are 1)."""
    frontier = deque([start])          # FIFO queue
    parent = {start: None}             # doubles as the visited set
    expanded = 0
    generated = 1
    while frontier:
        state = frontier.popleft()     # oldest state first
        expanded += 1                  # COUNT: expansion
        if state == goal:
            path = []
            while state is not None:
                path.append(state)
                state = parent[state]
            path.reverse()
            return SearchResult(True, path, len(path) - 1, expanded, expanded, generated)
        for _action, nxt in neighbours(grid, state):
            if nxt not in parent:      # mark visited when generated, so nothing is queued twice
                parent[nxt] = state
                frontier.append(nxt)
                generated += 1
    return SearchResult(False, None, None, expanded, expanded, generated)
