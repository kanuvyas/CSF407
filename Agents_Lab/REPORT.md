# Agents Lab: Goal-Based Warehouse Agent

Code is in `src/`, outputs in `results/`. Squares are written (row, col), counted from 0.

## Task 1 – Understanding the Problem

**1. Environment.** A fixed 7 x 21 grid of 66 non-wall squares. Shelving (`#`) blocks movement, `.` is free, `S` is the start and `G` the goal. It is static, fully known, deterministic and has a single agent.

**2. Goal.** Drive the vehicle from S (1,1) to G (1,19) without touching an obstacle, using as few moves as possible.

**3. Actions.** Up, Down, Left, Right. Each moves one square, and a move into a `#` or off the map is not allowed.

**4. Information the agent keeps.** Its current position, the map, and the goal position. While planning it also keeps the squares already visited and how it reached each one, so it can rebuild the path.

**5. Goal-based, not reflex.** A reflex agent chooses from the current percept only, for example "go Right if free". That rule gets stuck at (1,5), where the wall at (1,6) blocks it even though the goal is straight ahead. The goal-based agent searches ahead for a sequence of actions that reaches G, so it goes down and around the wall.

**Think About It.** The same strategy would still work, because BFS stays correct on any size of grid. The cost is what grows. I scaled each square into a k x k block (`results/scaling_results.txt`):

| k | Size | Free squares | Path length | BFS expanded | BFS largest frontier | A* expanded |
|---|------|-------------|-------------|--------------|----------------------|-------------|
| 1 | 7x21 | 66 | 20 | 59 | 5 | 23 |
| 2 | 14x42 | 264 | 40 | 237 | 10 | 80 |
| 4 | 28x84 | 1056 | 80 | 935 | 20 | 296 |
| 8 | 56x168 | 4224 | 160 | 3723 | 40 | 1136 |

Doubling the width and height gives four times the squares and about four times the nodes expanded. A* with the Manhattan heuristic expands about three times fewer nodes here, but it grows at the same rate. For much bigger warehouses I would expect memory and time to be the problem, and A* or a similar heuristic search to be worth using. Other difficulties I can think of, which I did not test: moving obstacles or other vehicles, a map that is only partly known, and different costs for different squares. Each of these would need replanning or a different algorithm.

## Task 2 – Designing the Agent

| Component | In this lab |
|-----------|-------------|
| Environment | The warehouse grid |
| Current state | The vehicle's (row, col) |
| Goal | Be on the G square |
| Actions | Up, Down, Left, Right |
| Decision making | BFS planner that returns a list of moves |

The diagram is in `results/agent_block_diagram.svg`. As text:

```
  ENVIRONMENT (grid) --percept: map + position--> [ Current state ] --+
        ^                                                            v
        |                                        [ BFS planner ] <-- [ Goal ]
        |                                                |
        +-------- action: move one square <-- [ Actions: U D L R ]
```

I decided to treat the map as a graph with four neighbours per square and unit cost per move. When two moves are equally good, the order Up, Down, Left, Right decides (see the tie-break note below).

## Task 3 – Prompt Engineering

The output of `src/warehouse_agent.py` is saved in `results/run_output.txt`. The path found has 20 moves:

```
#####################
#S***.#************G#
#.##****##########..#
#....##.............#
#.######.###.#.###..#
#........#..........#
#####################
```

BFS expanded 59 nodes and the largest frontier was 5. For the walled-in goal, the program printed "No path exists from S to G" after 63 expansions.

**1. Working on the first attempt?** Yes. Claude's first version ran without errors and gave a valid 20-move path, so no repair prompt was needed. The only failure in the whole build was a wrong expected value in one of the tests, which was fixed (see the LLM disclosure).

**2. How to improve the prompt.** The working program does not mean the prompt was precise. I would add: "shortest path", the move order for ties, the exact output format, a message for unreachable goals, and a request for tests. Without "shortest" the prompt allows depth-first search, which my tests show returns a 24-move path on this map.

**3. Algorithm.** Breadth-first search (BFS) with a visited set.

**4. Why BFS.** The prompt asks for a collision-free path on a grid where every move costs the same. BFS is complete and returns the fewest moves in that case, and it is simple to explain. I cannot see inside the LLM, so this is the reason Claude gave in the code and the output, and it fits the problem.

## Tests and validation

19 tests pass (`results/test_output.txt`). Key checks:

- The path replays without hitting a wall and ends at G.
- The length is 20, which I hand-computed: Manhattan distance 18 plus 2 for the wall detour.
- The length matches a separate relaxation-based shortest distance.
- There are exactly two shortest paths, and the agent returns the one that goes down at column 4.
- 300 seeded random maps give the same length as the reference, or None when unreachable.
- Edge cases: walled-in goal, start next to goal, already at goal, map with no border, bad input maps.

Five deliberately broken planners were also written (`src/broken_agents.py`). The checks reject all of them (`results/broken_versions.txt`):

| Broken version | Caught because |
|----------------|----------------|
| walks through walls | path hits obstacles |
| no visited set | search gives up, no plan found |
| Up and Down swapped | path hits obstacles, does not end at G |
| empty plan when unreachable | returns a plan for an unreachable goal |
| depth-first search | path is not shortest |

**Tie-break.** Two shortest paths exist. Because Down comes before Right in the action order, BFS reaches (2,4) first and returns the route that goes down at column 4. A different order would pick the other route with the same length.

## LLM disclosure

- Claude (an LLM) wrote the code, the tests, the broken versions, the scaling experiment, the diagram and this report, from my request and the lab specification. None of this was written by hand.
- Generated and run by Claude during the build: `warehouse_agent.py` was not modified after its first run. `scaling_experiment.py` had its free-square count simplified after the first run.
- Bug found in testing: a test Claude wrote, `test_single_corridor_forces_route`, expected 3 moves but the correct answer is 4 (`Right, Down, Down, Right`). The agent was right and the test was wrong, so the test was fixed.
- Decisions made: the tie-break order; a visited set so there are no loops; input validation (one S, one G, rectangular, known characters) for unseen maps; and `None` for "no path", kept separate from `[]` for "already at the goal".
- Independent checks: the hand-computed 20 and the two-path count were worked out from the map by reasoning before the code was run. The relaxation-based distance is a different method from BFS.

## Reflection on LLM-assisted engineering

The LLM was fast and its first program worked, which is a strength for a small, well-known problem like grid search. A limitation is that it never proved its own answer. Whether the path was shortest, or what the tie-break was, only became clear when it was checked against independent calculations. Broken versions were useful because they showed that the tests really detect wall-crossing, loops, wrong labels and non-shortest paths. I would not trust a generated agent on a larger or changing warehouse without tests like these.
