"""warehouse.py - the warehouse search problem (Lab Tasks 0 and 1).

Holds everything that describes the *problem* (not the search):
  - state       : a (row, col) tuple
  - actions     : Up, Down, Left, Right, each with cost 1
  - transition  : neighbours()
  - heuristics  : zero, Manhattan, Euclidean, k * Manhattan
  - SearchResult: what every search reports when it stops
"""
import math
from dataclasses import dataclass, field

# Each action is (name, (d_row, d_col)). Row 0 is the top of the map.
ACTIONS = [("Up", (-1, 0)), ("Down", (1, 0)), ("Left", (0, -1)), ("Right", (0, 1))]


@dataclass
class SearchResult:
    """Outcome of one search run."""
    found: bool                 # was a path to G found?
    path: list = field(default_factory=list)  # list of (row, col) from S to G, or None
    cost: int = None            # number of moves (each move costs 1)
    expanded: int = 0           # states removed from the frontier and expanded
    unique_expanded: int = 0    # distinct states among those (expanded > unique => re-opening)
    generated: int = 0          # states pushed onto the frontier


def parse_map(text):
    """Turn an ASCII map into (grid, start, goal). grid is a list of strings.

    Raises ValueError for ragged maps, unknown symbols, or a missing/duplicate S or G."""
    grid = [line.strip() for line in text.strip().splitlines()]
    if not grid or any(len(row) != len(grid[0]) for row in grid):
        raise ValueError("map must be non-empty and rectangular")
    start = goal = None
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch not in "#.SG":
                raise ValueError(f"unknown symbol {ch!r} at {(r, c)}")
            if ch == "S":
                if start is not None:
                    raise ValueError("more than one S")
                start = (r, c)
            if ch == "G":
                if goal is not None:
                    raise ValueError("more than one G")
                goal = (r, c)
    if start is None or goal is None:
        raise ValueError("map needs one S and one G")
    return grid, start, goal


def is_free(grid, cell):
    """A cell is usable if it is inside the map and not an obstacle."""
    r, c = cell
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] != "#"


def neighbours(grid, state):
    """Transition function: yield (action_name, next_state) for every VALID action.

    An action is invalid if it leaves the map or enters a '#' cell."""
    r, c = state
    for name, (dr, dc) in ACTIONS:
        nxt = (r + dr, c + dc)
        if is_free(grid, nxt):
            yield name, nxt


# ---- heuristics: h(state, goal) -> estimated remaining cost ----
def h_zero(state, goal):
    """h = 0: A* degenerates into uniform-cost search."""
    return 0


def h_manhattan(state, goal):
    """|x - xG| + |y - yG|: exact cost on an empty grid with 4-way moves."""
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def h_euclid(state, goal):
    """Straight-line distance; never larger than Manhattan, so still admissible."""
    return math.hypot(state[0] - goal[0], state[1] - goal[1])


def h_scaled(k):
    """Return the heuristic k * Manhattan (k > 1 is NOT admissible)."""
    def h(state, goal):
        return k * h_manhattan(state, goal)
    h.__doc__ = f"{k} x Manhattan distance"
    return h


def draw_path(grid, path):
    """Return the map as text with the path marked by '*' (S and G kept)."""
    rows = [list(row) for row in grid]
    for r, c in path[1:-1]:
        rows[r][c] = "*"
    return "\n".join("".join(row) for row in rows)
