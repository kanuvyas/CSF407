"""show_mutants.py - prints how many checks each deliberately broken A* fails (Task 3/7).

Run from the repository folder:   python3 tests/show_mutants.py
Saves results/mutant_report.txt."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from test_search import run_checks
from astar import astar
from broken_astar import astar_buggy, BUGS

lines = []
fails = run_checks(lambda g, s, t: astar(g, s, t))
lines.append(f"real A*          : {len(fails)} failed checks")
for bug in BUGS:
    fails = run_checks(lambda g, s, t, b=bug: astar_buggy(g, s, t, b))
    lines.append(f"{bug:<17}: {len(fails)} failed checks; first: {fails[0] if fails else '-'}")
text = "\n".join(lines)
print(text)
out = os.path.join(os.path.dirname(__file__), "..", "results", "mutant_report.txt")
open(out, "w").write(text + "\n")
