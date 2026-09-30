#!/usr/bin/env bash
# Reproduces all results in REPORT.md (needs Python 3, pytest; SWI-Prolog for the optional part)
cd "$(dirname "$0")"
mkdir -p results
python3 run_experiments.py | tee results/planner_output.txt
python3 -m pytest -q tests | tee results/pytest.txt
if command -v swipl >/dev/null; then ./run_prolog.sh | tee results/prolog_output.txt; else echo "swipl not installed, skipping Prolog"; fi
