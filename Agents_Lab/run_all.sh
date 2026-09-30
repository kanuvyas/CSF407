#!/bin/sh
# Re-creates everything in results/ from scratch. Run from the Agents_Lab folder:
#   sh run_all.sh
set -e
cd "$(dirname "$0")"
mkdir -p results
export PYTHONDONTWRITEBYTECODE=1
python3 src/warehouse_agent.py > results/run_output.txt
(cd src && python3 scaling_experiment.py) > results/scaling_results.txt
python3 tests/show_broken_catches.py > results/broken_versions.txt
# unittest prints its run time, which changes every run, so remove it
python3 -m unittest discover -s tests -v 2>&1 | sed 's/ in [0-9.]*s$//' > results/test_output.txt
tail -3 results/test_output.txt
