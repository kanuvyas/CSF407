"""
test_agent.py - Tests for the warehouse agent (Lab Task 3: validate the code).

Run:  python3 -m unittest discover -s tests -v     (from the Agents_Lab folder)

The tests do NOT trust the agent's own helpers. They use their own move table,
their own path replay and their own shortest-distance calculation (repeated
relaxation, a different method from BFS), plus a hand-computed answer for the
lab map. The last part checks that deliberately broken planners are caught.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from warehouse_agent import (BLOCKED_MAP, WAREHOUSE_MAP, GoalBasedAgent,
                             draw_path, parse_map)
from broken_agents import BROKEN
from scaling_experiment import scale_map

MOVES = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}


# ---------------------------------------------------------------------------
# Independent helpers (deliberately not reusing the agent's code)
# ---------------------------------------------------------------------------

def find(lines, ch):
    """Return the (row, col) of the single square containing ch."""
    for r, row in enumerate(lines):
        if ch in row:
            return (r, row.index(ch))


def replay(lines, plan):
    """Follow plan from S. Return the list of squares visited, or raise
    AssertionError if a move leaves the map or enters a '#'."""
    pos = find(lines, "S")
    trail = [pos]
    for action in plan:
        dr, dc = MOVES[action]
        pos = (pos[0] + dr, pos[1] + dc)
        assert 0 <= pos[0] < len(lines) and 0 <= pos[1] < len(lines[0]), "left the map"
        assert lines[pos[0]][pos[1]] != "#", "hit an obstacle at %s" % (pos,)
        trail.append(pos)
    return trail


def reference_distances(lines):
    """Distance from S to every free square by repeated relaxation
    (Bellman-Ford style sweeps). Unreachable squares are missing from the dict."""
    dist = {find(lines, "S"): 0}
    changed = True
    while changed:
        changed = False
        for (r, c), d in list(dist.items()):
            for dr, dc in MOVES.values():
                n = (r + dr, c + dc)
                if 0 <= n[0] < len(lines) and 0 <= n[1] < len(lines[0]) \
                        and lines[n[0]][n[1]] != "#" and dist.get(n, 10 ** 9) > d + 1:
                    dist[n] = d + 1
                    changed = True
    return dist


def count_shortest_paths(lines):
    """Number of different shortest S->G paths (dynamic programming by distance)."""
    dist = reference_distances(lines)
    goal = find(lines, "G")
    if goal not in dist:
        return 0
    ways = {find(lines, "S"): 1}
    for sq in sorted(dist, key=dist.get):           # process nearest squares first
        for dr, dc in MOVES.values():
            n = (sq[0] + dr, sq[1] + dc)
            if n in dist and dist[n] == dist[sq] + 1:
                ways[n] = ways.get(n, 0) + ways.get(sq, 0)
    return ways[goal]


def random_map(rng, rows=8, cols=8, wall_prob=0.3):
    """A random map with a wall border, S and G on different free squares."""
    grid = [["#"] * cols for _ in range(rows)]
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            grid[r][c] = "#" if rng.random() < wall_prob else "."
    inner = [(r, c) for r in range(1, rows - 1) for c in range(1, cols - 1)]
    s, g = rng.sample(inner, 2)
    grid[s[0]][s[1]] = "S"
    grid[g[0]][g[1]] = "G"
    return ["".join(row) for row in grid]


def problems_with(planner, n_random=30):
    """Run any planner (map lines -> action list or None) through the checks.
    Return a list of problems found; an empty list means the planner is good."""
    found = []
    maps = [WAREHOUSE_MAP, BLOCKED_MAP] + [random_map(random.Random(i)) for i in range(n_random)]
    for lines in maps:
        dist = reference_distances(lines)
        goal = find(lines, "G")
        plan = planner(lines)
        if goal not in dist:                        # goal unreachable
            if plan is not None:
                found.append("returned a plan for an unreachable goal")
            continue
        if plan is None:
            found.append("found no plan although one exists")
            continue
        try:
            trail = replay(lines, plan)
        except AssertionError as e:
            found.append("invalid plan: %s" % e)
            continue
        if trail[-1] != goal:
            found.append("plan does not end at G")
        elif len(plan) != dist[goal]:
            found.append("plan is not shortest (%d vs %d)" % (len(plan), dist[goal]))
    return sorted(set(found))


# ---------------------------------------------------------------------------
# 1. Tests on the lab map
# ---------------------------------------------------------------------------

class TestLabMap(unittest.TestCase):
    def setUp(self):
        self.agent = GoalBasedAgent(WAREHOUSE_MAP)
        self.plan = self.agent.plan()

    def test_length_is_hand_computed_20(self):
        # By hand: Manhattan distance S->G is 18; the wall at column 6 of row 1
        # forces a detour down and back up (+2), so 20 moves.
        self.assertEqual(len(self.plan), 20)

    def test_plan_is_collision_free_and_reaches_goal(self):
        trail = replay(WAREHOUSE_MAP, self.plan)
        self.assertEqual(trail[-1], find(WAREHOUSE_MAP, "G"))

    def test_no_square_visited_twice(self):
        trail = replay(WAREHOUSE_MAP, self.plan)
        self.assertEqual(len(trail), len(set(trail)))

    def test_length_matches_independent_distance(self):
        dist = reference_distances(WAREHOUSE_MAP)
        self.assertEqual(len(self.plan), dist[find(WAREHOUSE_MAP, "G")])

    def test_exactly_two_shortest_paths_and_agent_picks_one(self):
        # Hand check: the agent can go down at column 4 or at column 5 of row 1.
        self.assertEqual(count_shortest_paths(WAREHOUSE_MAP), 2)
        tail = ["Right"] * 12
        option_a = ["Right"] * 3 + ["Down"] + ["Right"] * 3 + ["Up"] + tail
        option_b = ["Right"] * 4 + ["Down"] + ["Right"] * 2 + ["Up"] + tail
        self.assertIn(self.plan, [option_a, option_b])
        self.assertEqual(self.plan, option_a)       # tie-break: Down tried before Right

    def test_deterministic(self):
        self.assertEqual(GoalBasedAgent(WAREHOUSE_MAP).plan(), self.plan)

    def test_execute_ends_at_goal(self):
        trail = self.agent.execute(self.plan)
        self.assertEqual(self.agent.state, self.agent.goal)
        self.assertEqual(trail[0], self.agent.start)

    def test_execute_rejects_a_collision(self):
        with self.assertRaises(ValueError):
            GoalBasedAgent(WAREHOUSE_MAP).execute(["Up"])   # wall above S

    def test_draw_path_marks_only_free_squares(self):
        trail = replay(WAREHOUSE_MAP, self.plan)
        drawn = draw_path(WAREHOUSE_MAP, trail)
        self.assertEqual(sum(row.count("*") for row in drawn), 19)  # 21 squares - S - G
        self.assertEqual(sum(row.count("#") for row in drawn),
                         sum(row.count("#") for row in WAREHOUSE_MAP))


# ---------------------------------------------------------------------------
# 2. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    def test_goal_walled_in_gives_none(self):
        self.assertIsNone(GoalBasedAgent(BLOCKED_MAP).plan())

    def test_start_is_goal_neighbour(self):
        self.assertEqual(GoalBasedAgent(["#####", "#SG.#", "#####"]).plan(), ["Right"])

    def test_open_map_without_border_stays_inside_grid(self):
        plan = GoalBasedAgent(["S..", "...", "..G"]).plan()
        self.assertEqual(len(plan), 4)
        replay(["S..", "...", "..G"], plan)

    def test_single_corridor_forces_route(self):
        plan = GoalBasedAgent(["S.#", "#.#", "#.G"]).plan()
        self.assertEqual(plan, ["Right", "Down", "Down", "Right"])

    def test_already_at_goal_gives_empty_plan(self):
        agent = GoalBasedAgent(["#####", "#SG.#", "#####"])
        agent.state = agent.goal
        self.assertEqual(agent.plan(), [])

    def test_bad_maps_are_rejected(self):
        for bad in ([], ["S.", "."], ["S.G", "..x"], ["S.."], ["SSG"], ["..G"], ["SGG"]):
            with self.assertRaises(ValueError, msg=str(bad)):
                parse_map(bad)


# ---------------------------------------------------------------------------
# 3. Property tests on seeded random maps and scaled maps
# ---------------------------------------------------------------------------

class TestProperties(unittest.TestCase):
    def test_random_maps_match_reference(self):
        rng = random.Random(407)                    # seeded, so always the same maps
        reachable = unreachable = 0
        for _ in range(300):
            lines = random_map(rng, rng.randint(4, 10), rng.randint(4, 10))
            plan = GoalBasedAgent(lines).plan()
            dist = reference_distances(lines)
            goal = find(lines, "G")
            if goal in dist:
                reachable += 1
                self.assertEqual(replay(lines, plan)[-1], goal)
                self.assertEqual(len(plan), dist[goal])
            else:
                unreachable += 1
                self.assertIsNone(plan)
        # make sure the sample really contained both kinds of map
        self.assertGreater(reachable, 30)
        self.assertGreater(unreachable, 30)

    def test_scaled_maps_length_grows_linearly(self):
        for k in (1, 2, 3, 4):
            lines = scale_map(WAREHOUSE_MAP, k)
            self.assertEqual(len(GoalBasedAgent(lines).plan()), 20 * k)


# ---------------------------------------------------------------------------
# 4. The checker must accept the real agent and reject every broken one
# ---------------------------------------------------------------------------

def real_planner(lines):
    """The real agent, wrapped as a planner function."""
    return GoalBasedAgent(lines).plan()


class TestBrokenVersionsAreCaught(unittest.TestCase):
    def test_real_agent_has_no_problems(self):
        self.assertEqual(problems_with(real_planner), [])

    def test_every_broken_version_is_caught(self):
        for name, planner in BROKEN.items():
            with self.subTest(bug=name):
                self.assertTrue(problems_with(planner), "tests missed bug: " + name)


if __name__ == "__main__":
    unittest.main()
