"""
show_broken_catches.py - Print what the checker says about each broken planner.

Lab: Task 3 (validating generated code). Output is saved to
results/broken_versions.txt. Run from the Agents_Lab folder:
    python3 tests/show_broken_catches.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from broken_agents import BROKEN
from test_agent import problems_with, real_planner


def kinds(planner):
    """The kinds of problem found, with map coordinates removed."""
    return sorted({p.split(":")[0].split(" (")[0]
                   for p in problems_with(planner)})


def main():
    """Print one line per planner: caught or not, and why."""
    print("%-28s %-8s %s" % ("planner", "caught?", "kinds of problem found"))
    print("%-28s %-8s %s" % ("real agent", "-", kinds(real_planner) or "none (good)"))
    for name, planner in BROKEN.items():
        found = kinds(planner)
        print("%-28s %-8s %s" % (name, "YES" if found else "NO", "; ".join(found)))


if __name__ == "__main__":
    main()
