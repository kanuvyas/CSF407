# Neural Models Lab: XOR, Depth, Activations and Output Layers

Lab exercise on learning, depth, activation functions and output layers, using the
"redundant safety sensors" XOR problem and a three-class extension.

## Layout

| Path | Contents |
|---|---|
| `src/common.py` | data, model builder (2-H-K MLP), full-batch training loop |
| `src/task1_linear_baseline.py` | Task 1: affine + sigmoid cannot learn XOR, plus the 4-point sketch |
| `src/task3_xor_binary.py` | Tasks 3 + 4A/4B: 2-2-1 tanh network, BCEWithLogitsLoss, gradient checks |
| `src/task4c_symmetry.py` | Task 4C: zero / identical initialisation |
| `src/task4d_activations.py` | Task 4D: sigmoid vs tanh vs ReLU (2 seeds + 40-seed summary) |
| `src/task5_three_class.py` | Task 5: 3 logits, cross-entropy, softmax checks, p - y |
| `tests/test_lab.py` | pytest checks for the claims in the report |
| `results/` | text logs, JSON summaries and figures produced by `run_all.sh` |
| `REPORT.md` | write-up: specification, design, results, reflection answers |
| `prompts.md` | LLM prompts and the corrections made to the generated code |
| `run_all.sh` | runs everything and the tests |

## Run

```bash
pip install -r requirements.txt     # CPU PyTorch is enough
./run_all.sh                        # about a minute; regenerates everything in results/
```

All experiments use fixed seeds, so the numbers in `REPORT.md` are reproducible.

## Headline results

* Linear model on XOR: stuck at loss ln 2 = 0.693, every probability about 0.5.
* 2-2-1 tanh network (seed 0): loss 0.831 -> 0.0006, all 4 predictions correct.
* Zero initialisation: hidden rows identical at every step, loss never leaves 0.693.
* Activations (seed 0): sigmoid and tanh solve it, ReLU gets stuck. Over 40 seeds only 30/40, 15/40
  and 8/40 runs succeed for sigmoid, tanh and ReLU: a 2-unit XOR network is very sensitive to its
  initialisation.
* Three-class extension: all 4 inputs correct, softmax sums to 1, logit gradient equals p - y.
