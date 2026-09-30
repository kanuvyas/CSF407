"""test_search.py - tests for the A* / BFS warehouse agent (Lab Task 3 and 7).

Run from the repository folder:   python3 -m unittest discover -s tests -v

Idea: check properties that must hold for a correct implementation
(hand-computed values, path validity, optimality against an independent BFS,
admissibility) and then confirm that deliberately broken versions FAIL those checks."""
import os
import random
import sys
import unittest
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import maps
from warehouse import (parse_map, neighbours, h_zero, h_manhattan, h_euclid, h_scaled)
from astar import astar
from bfs import bfs
from broken_astar import astar_buggy, BUGS


# ---------------------------------------------------------------- helpers
def random_map(rng, rows=8, cols=8, p_wall=0.3):
    """Random walled map with S and G on distinct free cells (seeded rng => repeatable)."""
    cells = [[("#" if (r in (0, rows - 1) or c in (0, cols - 1) or rng.random() < p_wall) else ".")
              for c in range(cols)] for r in range(rows)]
    free = [(r, c) for r in range(rows) for c in range(cols) if cells[r][c] == "."]
    s, g = rng.sample(free, 2)
    cells[s[0]][s[1]] = "S"
    cells[g[0]][g[1]] = "G"
    return "\n".join("".join(row) for row in cells)


def true_distances(grid, source):
    """Independent BFS from `source` to every reachable cell (used as ground truth h*)."""
    dist = {source: 0}
    q = deque([source])
    while q:
        cur = q.popleft()
        for _, nxt in neighbours(grid, cur):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


def path_problems(grid, start, goal, res):
    """Return a list of reasons why res.path is not a valid path (empty list = valid)."""
    bad = []
    p = res.path
    if p[0] != start:
        bad.append("path does not start at S")
    if p[-1] != goal:
        bad.append("path does not end at G")
    for a, b in zip(p, p[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            bad.append(f"non-adjacent step {a}->{b}")
        if grid[b[0]][b[1]] == "#":
            bad.append(f"path enters wall at {b}")
    if len(set(p)) != len(p):
        bad.append("path repeats a cell")
    if res.cost != len(p) - 1:
        bad.append(f"reported cost {res.cost} != moves {len(p) - 1}")
    return bad


def run_checks(solver, n_random=150):
    """Apply every check to solver(grid, start, goal) -> SearchResult.

    Returns a list of failure messages; an empty list means the solver looks correct."""
    fails = []
    # hand-computed cases
    g, s, t = parse_map(maps.WAREHOUSE)
    r = solver(g, s, t)
    if not (r.found and r.cost == 40):        # 16 + 2 + 6 + 2 + 8 + 6 traced by hand
        fails.append(f"warehouse: expected cost 40, got {r.cost}")
    g, s, t = parse_map(maps.TRIVIAL)
    r = solver(g, s, t)
    if not (r.found and r.cost == 1):
        fails.append(f"trivial: expected cost 1, got {r.cost}")
    g, s, t = parse_map(maps.NO_SOLUTION)
    r = solver(g, s, t)
    if r.found:
        fails.append("no-solution map: reported a solution")
    g, s, t = parse_map(maps.ALTERNATIVE)
    r = solver(g, s, t)
    if not (r.found and r.cost == 4):
        fails.append(f"alternative: expected cost 4, got {r.cost}")
    g, s, t = parse_map(maps.HEURISTIC_TRAP)
    r = solver(g, s, t)
    if not (r.found and r.cost == 6):          # hand-computed: R,R,U,R,R,U
        fails.append(f"heuristic trap: expected cost 6, got {r.cost}")
    g, s, t = parse_map("#####\n#.S.#\n#.G.#\n#####")
    r = solver(g, s, t)
    if not (r.found and r.cost == 1):
        fails.append("vertical neighbour: expected cost 1")
    # random maps compared with an independent BFS
    rng = random.Random(407)
    for i in range(n_random):
        g, s, t = parse_map(random_map(rng))
        truth = true_distances(g, s).get(t)
        r = solver(g, s, t)
        free = sum(row.count(".") + row.count("S") + row.count("G") for row in g)
        if r.found != (truth is not None):
            fails.append(f"random {i}: found={r.found} but reachable={truth is not None}")
            continue
        if r.found:
            fails += [f"random {i}: {m}" for m in path_problems(g, s, t, r)]
            if r.cost != truth:
                fails.append(f"random {i}: cost {r.cost} != shortest {truth}")
        if r.expanded > free:
            fails.append(f"random {i}: expanded {r.expanded} > {free} free cells")
    return fails


# ---------------------------------------------------------------- tests
class TestProblem(unittest.TestCase):
    def test_parse_positions(self):
        g, s, t = parse_map(maps.WAREHOUSE)
        self.assertEqual((s, t), ((1, 1), (7, 15)))

    def test_valid_actions_at_start(self):
        g, s, t = parse_map(maps.WAREHOUSE)
        self.assertEqual([a for a, _ in neighbours(g, s)], ["Down", "Right"])  # Up/Left are walls

    def test_bad_maps_rejected(self):
        for bad in ["#S#", "#SG#\n#.#", "#SGG#", "#S?G#", "#S..#"]:
            with self.assertRaises(ValueError):
                parse_map(bad)

    def test_edge_of_map_is_not_free(self):
        g, s, t = parse_map("SG\n..")            # no border walls: moving off the map must be invalid
        self.assertEqual({a for a, _ in neighbours(g, s)}, {"Right", "Down"})


class TestAStar(unittest.TestCase):
    def test_all_checks_pass_for_real_astar(self):
        self.assertEqual(run_checks(lambda g, s, t: astar(g, s, t)), [])

    def test_bfs_passes_same_checks(self):
        self.assertEqual(run_checks(bfs), [])

    def test_trivial_path_exact(self):
        g, s, t = parse_map(maps.TRIVIAL)
        r = astar(g, s, t)
        self.assertEqual(r.path, [(1, 1), (1, 2)])

    def test_start_equals_goal_cost_zero(self):
        g, s, t = parse_map("#S.G#")             # any valid grid; then use S as the goal too
        r = astar(g, s, s)
        self.assertTrue(r.found)
        self.assertEqual((r.cost, r.path, r.expanded), (0, [s], 1))

    def test_no_solution_terminates_with_failure(self):
        g, s, t = parse_map(maps.NO_SOLUTION)
        r = astar(g, s, t, max_expansions=10_000)   # cap would show up as expanded > free cells
        self.assertFalse(r.found)
        self.assertIsNone(r.path)
        self.assertEqual(r.unique_expanded, 9)      # the 9 cells reachable from S, counted by hand

    def test_open_room_many_paths_still_shortest(self):
        g, s, t = parse_map(maps.OPEN_ROOM)
        r = astar(g, s, t)
        self.assertEqual(r.cost, 4)
        self.assertEqual(path_problems(g, s, t, r), [])

    def test_manhattan_never_reopens(self):
        """Manhattan is consistent, so no state should be expanded twice."""
        rng = random.Random(1)
        for _ in range(100):
            g, s, t = parse_map(random_map(rng))
            r = astar(g, s, t)
            self.assertEqual(r.expanded, r.unique_expanded)

    def test_optimal_heuristics_agree_with_bfs(self):
        rng = random.Random(2)
        for _ in range(100):
            g, s, t = parse_map(random_map(rng))
            want = bfs(g, s, t).cost
            for h in (h_zero, h_manhattan, h_euclid):
                self.assertEqual(astar(g, s, t, h).cost, want)

    def test_scaled_heuristic_is_suboptimal_on_trap_map(self):
        """Documents the Task 6 finding: 2 x Manhattan returns 8 moves, not 6."""
        g, s, t = parse_map(maps.HEURISTIC_TRAP)
        self.assertEqual(astar(g, s, t, h_manhattan).cost, 6)
        self.assertEqual(astar(g, s, t, h_scaled(2)).cost, 8)

    def test_astar_no_more_expansions_than_bfs_on_warehouse(self):
        g, s, t = parse_map(maps.WAREHOUSE)
        self.assertLessEqual(astar(g, s, t).expanded, bfs(g, s, t).expanded)


class TestHeuristics(unittest.TestCase):
    def test_admissibility(self):
        """h(n) <= true remaining cost for Manhattan and Euclidean; 2x Manhattan must
        violate it somewhere (otherwise the 'not admissible' claim would be wrong)."""
        rng = random.Random(3)
        violated = {"manhattan": False, "euclid": False, "double": False}
        for _ in range(100):
            g, s, t = parse_map(random_map(rng))
            dist = true_distances(g, t)              # distance to goal = distance from goal
            for cell, d in dist.items():
                violated["manhattan"] |= h_manhattan(cell, t) > d
                violated["euclid"] |= h_euclid(cell, t) > d + 1e-9
                violated["double"] |= h_scaled(2)(cell, t) > d
        self.assertFalse(violated["manhattan"])
        self.assertFalse(violated["euclid"])
        self.assertTrue(violated["double"])


class TestBrokenVersionsAreCaught(unittest.TestCase):
    def test_reference_is_clean(self):
        self.assertEqual(run_checks(lambda g, s, t: astar(g, s, t)), [])

    def test_each_bug_is_detected(self):
        for bug in BUGS:
            with self.subTest(bug=bug):
                fails = run_checks(lambda g, s, t, b=bug: astar_buggy(g, s, t, b))
                self.assertGreater(len(fails), 0, f"bug {bug!r} slipped through the tests")


if __name__ == "__main__":
    unittest.main()
