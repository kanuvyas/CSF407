# Search Lab: A* and BFS on the Warehouse Map

Implements A* (and BFS for comparison) for the warehouse robot problem, tests it, and
investigates the heuristic. Python 3 standard library only.

**LLM disclosure:** the search code, tests and this write-up were produced with Claude from my
specification (see `prompts.md` and REPORT.md, Task 7).

## Deliverables

| Deliverable (handout section 5) | File |
|---|---|
| 1. Formulation of the search problem | `REPORT.md`, Task 0 |
| 2. Design of the agent | `REPORT.md` Task 1, `prompts.md` section A |
| 3. Final Python program | `src/astar.py`, `src/bfs.py`, `src/warehouse.py`, `src/maps.py`, `src/run_lab.py` |
| 4. Prompts used with the LLM | `prompts.md` section C |
| 5. Test results | `REPORT.md` Task 3, `results/test_output.txt`, `results/mutant_report.txt`, `tests/test_search.py` |
| 6. BFS/A* comparison | `REPORT.md` Task 5, `results/main_output.txt` |
| 7. Heuristic investigation | `REPORT.md` Task 6, `results/main_output.txt` |
| 8. Reflection answers | `REPORT.md` Task 7 and Final Reflection |

## How to run (from this folder)

```text
python3 src/run_lab.py                              # all experiments -> results/main_output.txt, results.json
python3 -m unittest discover -s tests -v            # 17 tests
python3 tests/show_mutants.py                       # which checks catch each broken version
```

## Main results

- Warehouse: 40 moves for BFS and A* (all Manhattan/Euclidean/h=0/scaled variants too).
- BFS and A* both expand all 64 free cells on this map; on an extra open-floor map A* expands 21 vs BFS 89.
- 2 x and 5 x Manhattan return 8 moves instead of 6 on a small trap map (Task 6).
- The tests pass; five deliberately broken A* versions are all caught.
