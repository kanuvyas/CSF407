# Search Lab Report: A* on the Warehouse Map

Everything below comes from `results/` (`main_output.txt`, `results.json`, `test_output.txt`, `mutant_report.txt`). Counting rule: "states expanded" = every non-stale state removed from the frontier, the goal included.

## Task 0: Search problem

| Component | Specification |
|---|---|
| State S | robot position `(row, col)` on a free cell |
| Actions A | Up, Down, Left, Right |
| Transition T | `(r,c)` moves by the action's step if the new cell is on the map and not `#` |
| Initial state s0 | `(1,1)` (the S cell) |
| Goal G | `{(7,15)}` |
| Cost c | 1 per move |

(a) Only the robot's position; the walls never change, so they are part of the problem, not the state. (b) An action is invalid if it leaves the map or enters a `#` cell. (c) Yes, deterministic: one action from one state always gives the same next state. (d) A sequence of valid moves from `(1,1)` to `(7,15)`; an optimal solution has the fewest moves.

## Task 1: Design

My design table is in `prompts.md`, section A (state tuple, list-of-strings map, heap frontier ordered by `(f, h, counter)`, `best_g` dict, `parent` dict, goal test on removal). It reports found, path, length and states expanded. (Drafted with Claude, before any code was run.)

## Task 2: LLM generation

The prompts are in `prompts.md`, section C. Claude generated `src/astar.py` (and `src/bfs.py`) from my specification; `src/warehouse.py` holds the problem definition and heuristics.

## Task 3: Tests

| Test | Result |
|---|---|
| 1 Original warehouse | found, 40 moves, 64 expanded (the 40-move path is in `results/main_output.txt`) |
| 2 Trivial `#SG##` | found, path `(1,1)->(1,2)`, 1 move, 2 expanded |
| 3 No solution | found = False, 9 expanded (the 9 cells reachable from S), terminated |
| 4a Two routes (4 and 8 moves) | 4 moves, BFS also 4 |
| 4b Open room, many equal routes | 4 moves, BFS also 4 |

I hand-traced Test 1 before running: 16 + 2 + 6 + 2 + 8 + 6 = 40 moves, which matched. The unit tests (`tests/test_search.py`, 17 tests, all pass) also compare A* with an independent BFS on 150 seeded random maps: same reachability, same optimal cost, valid paths, no more expansions than free cells.

Five deliberately broken A* versions (`src/broken_astar.py`) must be caught by the checks:

| Bug | Failed checks | First symptom |
|---|---|---|
| no closed set | 49 | never finishes the warehouse within the cap of 3000 expansions |
| ignores g | 4 | warehouse cost 48, not 40 |
| 3 x Manhattan | 1 | trap map cost 8, not 6 |
| walks through walls | 153 | warehouse cost 20 |
| off-by-one length | 255 | cost 41, not 40 |

## Task 4: Inspecting the code

| Concept | Where |
|---|---|
| State | `(row, col)` tuples; start pushed at `astar.py:22` |
| Action | `ACTIONS`, `warehouse.py:14` |
| Transition | `neighbours()`, `warehouse.py:59` (bounds/wall check in `is_free`, line 53) |
| Goal test | `astar.py:38` |
| g(n) | `astar.py:48`, stored in `best_g` (line 50) |
| h(n) | called at `astar.py:53`; Manhattan defined at `warehouse.py:76` |
| f(n) | `astar.py:54` (`new_g + hn2`); for the start, line 22 |
| Frontier | `frontier` heap, `astar.py:22`, pop at line 30, push at line 54 |
| Visited states | `best_g`, `astar.py:23`, checked at lines 31 and 49 |
| Path reconstruction | `parent` dict (line 24), walked back at lines 40-43 |

(a) A binary heap (`heapq`) holding tuples `(f, h, counter, g, state)`. (b) `heappop` (line 30) returns the smallest tuple: lowest f, then lowest h, then oldest. (c) Line 53 calls `h`, which is `h_manhattan` by default. (d) Yes, f is computed explicitly when a state is pushed (line 54). (e) A state is only pushed again if a strictly cheaper g is found (line 49), and old heap entries are skipped when popped (line 31).

## Task 5: BFS vs A*

| Measure | BFS | A* (Manhattan) |
|---|---|---|
| Solution found | yes | yes |
| Path length | 40 | 40 |
| States expanded | 64 | 64 |

(a) Yes, both. (b) Yes, both 40 moves (and the same path). (c) Neither: both expanded all 64 free cells. My prediction that A* would expand a bit fewer was wrong. The start has h = 20 but the true cost is 40, and the dead-end bottom corridor has f = 20 at its end (by hand: g = 18, h = 2), below 40, so A* with a consistent heuristic has to expand it. (d) A* expands fewer states when h steers the search towards the goal. On the extra open-floor map (`OPEN_WAREHOUSE`, not in the handout) BFS expanded 89 states and A* 21, both finding 20 moves.

## Task 6: Heuristics

Manhattan is appropriate because every move changes row or column by exactly 1, so at least |dr| + |dc| moves are needed; walls can only add more (admissible, and consistent). Experiments:

| Heuristic | Warehouse: length / expanded | Open floor: length / expanded | Trap map: length / expanded |
|---|---|---|---|
| BFS (reference) | 40 / 64 | 20 / 89 | 6 / 12 |
| h = 0 | 40 / 64 | 20 / 89 | 6 / 12 |
| Manhattan | 40 / 64 | 20 / 21 | 6 / 11 |
| Euclidean | 40 / 64 | 20 / 68 | 6 / 9 |
| 2 x Manhattan | 40 / 68 | 20 / 21 | **8** / 9 |
| 5 x Manhattan (extra) | 40 / 84 | 20 / 21 | **8** / 9 |

The trap map (10 free cells, found by a seeded random search) is my own addition. On the warehouse, 2 x and 5 x Manhattan still found 40 but expanded 68 and 84 times for only 64 distinct states, meaning some states were re-opened when a cheaper route appeared. On the trap map they return 8 moves instead of 6. So h = 0 and Euclidean are safe but explore more; an overestimating h explores less but can lose optimality. Two predictions were wrong: 2 x Manhattan did not expand fewer states on the warehouse, and Euclidean expanded fewer (9) than Manhattan (11) on the trap map.

## Task 7: Evaluating the LLM

1. The generated `astar.py` and `bfs.py` passed every test the first time they were run; I have not edited them.
2. I found no bug in the search code. The problems I found were in my own checks: my expected count for Test 3 was 8 (the real value is 9), one test built a map with no G, and the first tests did not catch the 3 x Manhattan bug.
3. By running the tests: the mistakes above showed up as failures, and one broken version hung on path reconstruction because its parent links formed a loop, which I fixed by giving each entry its own parent chain.
4. Terms to make sure I understand: lazy deletion of stale heap entries, re-opening, admissible versus consistent.
5. No change to the generated search code; I only wrote the tests, broken versions and run script around it.
6. The comparison with an independent BFS on random maps, the hand-computed 40 and 6, and the trap map.
7. No. The 3 x Manhattan bug passed my first round of tests and still returns a plausible path.
8. Informed does not mean faster: on the warehouse A* does the same work as BFS, and the heuristic's quality decides the savings.

| | Designed by me | LLM suggested | Accepted | Changed | Tested |
|---|---|---|---|---|---|
| Problem formulation, design table | yes (drafted with Claude) | | yes | | |
| `astar.py`, `bfs.py` | | yes | yes | no | all tests |
| Tests, broken versions, extra maps | | yes | yes | fixed 3 mistakes (above) | run |

## Final Reflection

1. Formulating first forces me to fix what a state, action and cost are, so the code is a translation of a clear spec and the tests have something to check against.
2. A* is informed because it uses h, an estimate of the remaining cost, to choose what to expand next, where BFS only uses depth.
3. The heuristic decides both the work and the answer: h = 0 wastes effort, an admissible h keeps the result optimal, and an overestimating h (trap map) can return a longer path.
4. The LLM wrote the initial code, the tests and the broken versions from my specification; the design, the counting rule and the checking of the results were my responsibility.
5. Code that looks right can hide errors, like a heuristic that returns a valid path 8 moves long when 6 exist; only tests against known answers reveal it.
