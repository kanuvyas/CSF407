"""run_lab.py - runs every experiment of the lab and saves the output.

Run from the repository folder:   python3 src/run_lab.py
Writes results/main_output.txt and results/results.json. There is no randomness here,
so two runs give identical files.

Sections match the lab: 0/1 formulation, 3 tests, 5 BFS vs A*, 6 heuristics."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from warehouse import (parse_map, neighbours, draw_path, h_zero, h_manhattan,
                       h_euclid, h_scaled)
from astar import astar
from bfs import bfs
import maps

OUT = []          # every printed line is also collected here and saved to a file
RESULTS = {}      # machine-readable results


def say(text=""):
    """Print and remember a line of output."""
    print(text)
    OUT.append(text)


def report(label, res, grid=None):
    """Print the four required measures (plus extras) for one search run."""
    say(f"[{label}]")
    say(f"  solution found : {res.found}")
    say(f"  path length    : {res.cost}")
    say(f"  states expanded: {res.expanded} (distinct: {res.unique_expanded}, generated: {res.generated})")
    if res.found:
        say(f"  path (row,col) : {res.path}")
        if grid is not None:
            say("  map with path:")
            for line in draw_path(grid, res.path).splitlines():
                say("    " + line)
    RESULTS[label] = {"found": res.found, "length": res.cost, "expanded": res.expanded,
                      "distinct_expanded": res.unique_expanded, "generated": res.generated,
                      "path": res.path}


# ------------------------------------------------------------ Section 0/1
say("=== Section 0/1: the warehouse problem ===")
grid, start, goal = parse_map(maps.WAREHOUSE)
free = sum(row.count(".") + row.count("S") + row.count("G") for row in grid)
say(f"map size {len(grid)} x {len(grid[0])}, start {start}, goal {goal}, free cells (incl. S,G): {free}")
say("valid actions from S: " + ", ".join(a for a, _ in neighbours(grid, start)))
RESULTS["free_cells"] = free

# ------------------------------------------------------------ Section 3 tests
say("\n=== Section 3: Tests 1-4 (A*, Manhattan) ===")
say("--- Test 1: original warehouse ---")
report("test1_warehouse", astar(grid, start, goal), grid)

say("\n--- Test 2: trivial (goal adjacent) ---")
g2, s2, t2 = parse_map(maps.TRIVIAL)
report("test2_trivial", astar(g2, s2, t2), g2)

say("\n--- Test 3: no solution ---")
g3, s3, t3 = parse_map(maps.NO_SOLUTION)
report("test3_no_solution", astar(g3, s3, t3), g3)

say("\n--- Test 4a: alternative paths (top = 4 moves, bottom = 8 moves) ---")
g4, s4, t4 = parse_map(maps.ALTERNATIVE)
r4 = astar(g4, s4, t4)
report("test4a_alternative", r4, g4)
say(f"  BFS shortest length on this map: {bfs(g4, s4, t4).cost}")

say("\n--- Test 4b: open room, many equal shortest paths ---")
g5, s5, t5 = parse_map(maps.OPEN_ROOM)
r5 = astar(g5, s5, t5)
report("test4b_open_room", r5, g5)
say(f"  BFS shortest length on this map: {bfs(g5, s5, t5).cost}")

# ------------------------------------------------------------ Section 5
say("\n=== Section 5: BFS vs A* on the warehouse ===")
r_bfs = bfs(grid, start, goal)
r_astar = astar(grid, start, goal)
report("task5_bfs", r_bfs, grid)
report("task5_astar", r_astar)

# ------------------------------------------------------------ Section 6
say("\n=== Section 6: heuristic investigation (A*, same warehouse) ===")
heuristics = [
    ("h = 0", h_zero),
    ("Manhattan", h_manhattan),
    ("Euclidean", h_euclid),
    ("2 x Manhattan", h_scaled(2)),
    ("5 x Manhattan (extra)", h_scaled(5)),
]
for name, h in heuristics:
    report("task6_" + name, astar(grid, start, goal, h), grid if "extra" in name else None)

# ------------------------------------------------------------ Section 6b
say("\n=== Section 6b: heuristics on the small trap map (extra) ===")
gt, st, tt = parse_map(maps.HEURISTIC_TRAP)
report("trap_bfs", bfs(gt, st, tt))
for name, h in heuristics:
    report("trap_" + name, astar(gt, st, tt, h), gt if name == "2 x Manhattan" else None)

# ------------------------------------------------------------ Extra: open warehouse
say("\n=== Extra: open warehouse (not in the handout) ===")
go, so, to = parse_map(maps.OPEN_WAREHOUSE)
report("extra_open_bfs", bfs(go, so, to), go)
for name, h in heuristics:
    report("extra_open_astar_" + name, astar(go, so, to, h))

os.makedirs(os.path.join(os.path.dirname(__file__), "..", "results"), exist_ok=True)
res_dir = os.path.join(os.path.dirname(__file__), "..", "results")
with open(os.path.join(res_dir, "main_output.txt"), "w") as f:
    f.write("\n".join(OUT) + "\n")
with open(os.path.join(res_dir, "results.json"), "w") as f:
    json.dump(RESULTS, f, indent=1, sort_keys=True)
