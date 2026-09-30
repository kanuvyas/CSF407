# Logical Reasoning for Planning (Logic + Search = Planning)

| File | Contents |
|---|---|
| `planner.py` | action/state representation, `applicable`, `apply_action`, BFS planner, independent plan validator, warehouse problem |
| `tests/test_planner.py` | Tests A, B, C and a few extra checks (pytest) |
| `run_experiments.py` | prints every result used in the report |
| `planner.pl`, `run_prolog.sh` | optional Prolog extension (Tasks 6-8) and the queries run against it |
| `run_all.sh` | runs everything and saves output to `results/` |
| `REPORT.md` | specification, manual plan, test results, answers, reflections |
| `prompts.md` | the LLM prompt and notes on what was generated / changed |

Run: `./run_all.sh` (Python 3 + pytest; SWI-Prolog optional).
