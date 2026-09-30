# Bayesian Networks and Autoregressive Language Models (AI Laboratory)

A first-order (bigram) and a second-order (trigram) autoregressive language model,
built as Bayesian networks in plain Python. No machine-learning library and no
pretrained model. The training data is the six sentences from the handout.

## Folder layout

| Path | What it is |
|---|---|
| `REPORT.md` | **Main document.** Answers to Q1-Q14, the model comparison, and the LLM reflection |
| `prompts.md` | The specifications given to the LLM (Parts V, X, XII) and what was expected |
| `src/data.py` | training sentences, `<START>` / `<END>` |
| `src/first_order_lm.py` | first-order model `P(X_t \| X_{t-1})` |
| `src/second_order_lm.py` | second-order model `P(X_t \| X_{t-2}, X_{t-1})` |
| `src/run_lab.py` | runs every experiment and writes `results/` |
| `tests/test_models.py` | 49 property tests (see below) |
| `results/` | CPTs, normalisation check, predictions, generated text, greedy vs sampling, comparison, test log |

## Handout Section 21: deliverables

| # | Required | Where |
|---|---|---|
| 1 | first-order implementation | `src/first_order_lm.py` |
| 2 | second-order implementation | `src/second_order_lm.py` |
| 3 | conditional probability tables | `results/cpt_first_order.md`, `results/cpt_second_order.md`, `REPORT.md` Q3 |
| 4 | generated text | `results/generated_first_order.txt`, `results/generated_second_order.txt`, `results/greedy_vs_sampling.txt` |
| 5 | normalisation test results | `results/normalisation_tests.txt`, `results/test_output.txt` |
| 6 | answers to Questions 1-14 | `REPORT.md` |
| 7 | reflection on LLM use (with an inspected code example) | `REPORT.md`, last section; `prompts.md` |

## How to run

Python 3.8+ only, nothing to install.

```bash
python src/run_lab.py                       # regenerates everything in results/ (seed 42)
python tests/test_models.py                 # 49 tests
python -m unittest discover -s tests -v     # same tests through unittest
```

## What the tests check

| Group | Property |
|---|---|
| Data | six lower-case sentences, 10-word vocabulary, special tokens only added by the models |
| Counting | 42 transitions / 42 triples, hand counts, the handout's 3/5 and 2/5 example |
| Normalisation | every CPT row sums to 1, all probabilities in (0, 1], `<END>` is never a context |
| Prediction | arg max, alphabetical tie-break, unseen words never crash or add rows |
| Sampling | sampled frequencies match the CPT (total-variation distance < 0.01), greedy is deterministic |
| Chain rule | sentence probabilities equal hand products; total mass of the second-order model is exactly 1 |
| First vs second order | summing triple counts gives the pair counts; second-order support lies inside first-order support |
| Bug detection | four deliberately broken models must be caught (wrong denominator, lost `<END>`, uniform sampling, single `<START>` pad) |
