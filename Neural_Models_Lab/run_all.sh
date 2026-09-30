#!/usr/bin/env bash
# Reproduces every number in REPORT.md. CPU only, takes about a minute.
set -e
cd "$(dirname "$0")"
mkdir -p results
python3 src/task1_linear_baseline.py | tee results/task1_linear_baseline.txt
python3 src/task3_xor_binary.py      | tee results/task3_xor_binary.txt
python3 src/task4c_symmetry.py       | tee results/task4c_symmetry.txt
python3 src/task4d_activations.py    | tee results/task4d_activations.txt
python3 src/task5_three_class.py     | tee results/task5_three_class.txt
python3 -m pytest -q tests           | tee results/pytest.txt
