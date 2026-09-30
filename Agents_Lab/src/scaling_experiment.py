"""
scaling_experiment.py - "Think About It": what if the warehouse were bigger?

Lab: Task 1, Think About It.
Each square of the lab map is replaced by a k x k block (k = 1, 2, 4, 8) and
the search is repeated. I count nodes expanded for BFS and, for comparison, A*
with the Manhattan-distance heuristic. No timings are printed so the output is
identical on every run.

Run:  python3 src/scaling_experiment.py
"""

import heapq

from warehouse_agent import WAREHOUSE_MAP, GoalBasedAgent


def scale_map(lines, k):
    """Make every square a k x k block. S and G stay single squares (top-left)."""
    out = []
    for row in lines:
        for dr in range(k):
            line = ""
            for ch in row:
                if ch in "SG" and dr == 0:
                    line += ch + "." * (k - 1)      # marker only in the corner
                elif ch in "SG":
                    line += "." * k
                else:
                    line += ch * k
            out.append(line)
    return out


def astar(agent):
    """A* with Manhattan heuristic. Returns (path length or None, nodes expanded)."""
    def h(s):
        return abs(s[0] - agent.goal[0]) + abs(s[1] - agent.goal[1])

    best_cost = {agent.start: 0}
    heap = [(h(agent.start), 0, agent.start)]       # (f, g, state)
    expanded = 0
    while heap:
        f, g, state = heapq.heappop(heap)
        if g > best_cost[state]:
            continue                                # stale queue entry
        expanded += 1
        if agent.goal_test(state):
            return g, expanded
        for _, nxt in agent.successors(state):
            if g + 1 < best_cost.get(nxt, 10 ** 9):
                best_cost[nxt] = g + 1
                heapq.heappush(heap, (g + 1 + h(nxt), g + 1, nxt))
    return None, expanded


def main():
    """Print a table of search effort against warehouse size."""
    print("k = each original square becomes a k x k block")
    print("%-3s %-9s %-10s %-9s %-12s %-13s %-12s" % (
        "k", "size", "free sq.", "BFS len", "BFS expanded", "BFS frontier", "A* expanded"))
    for k in (1, 2, 4, 8):
        agent = GoalBasedAgent(scale_map(WAREHOUSE_MAP, k))
        free = sum(len(row) - row.count("#") for row in agent.grid)  # non-wall squares
        plan = agent.plan()
        a_len, a_exp = astar(agent)
        assert len(plan) == a_len, "BFS and A* must agree on the shortest length"
        print("%-3d %-9s %-10d %-9d %-12d %-13d %-12d" % (
            k, "%dx%d" % (len(agent.grid), len(agent.grid[0])), free, len(plan),
            agent.nodes_expanded, agent.max_frontier, a_exp))


if __name__ == "__main__":
    main()
