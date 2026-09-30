"""
run_lab.py -- runs every experiment of the lab and saves the outputs in ../results/.

Usage:   python src/run_lab.py

Sections (they follow the parts of the lab handout):
  1. build both models from the training data
  2. Q3   : conditional probability tables (CPTs)
  3. Part VII : normalisation test  (sum_v P(v|w) must be 1)
  4. Part VIII: next-word predictions and arg max
  5. Part IX  : generate 20 sentences with each model
  6. Part X   : greedy (Mode A) vs sampling (Mode B)
  7. Part XIII: comparison of first-order and second-order models
"""
import os
import random

from data import START, END, load_sentences
from first_order_lm import FirstOrderLM
from second_order_lm import SecondOrderLM

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
SEED = 42        # fixed random seed so results can be reproduced exactly
MAX_LEN = 20     # maximum sentence length (stops endless loops in greedy mode)


# ---------- small helper functions -------------------------------------------
def write(name, text):
    """Save text to results/<name>."""
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text)


def fmt_dist(d):
    """Format {word: prob} as 'word: 0.250, ...'."""
    return ", ".join(f"{w}: {p:.3f}" for w, p in d.items()) or "(no observed transition)"


def sent(words):
    """Turn a list of words into a printable sentence."""
    return " ".join(words) if words else "(empty)"


def gen_many(model, n, mode, seed):
    """Generate n sentences from a model with a seeded random generator."""
    rng = random.Random(seed)
    return [model.generate(mode, rng, MAX_LEN) for _ in range(n)]


# =============================================================================
# 1. Build both models
# =============================================================================
sentences = load_sentences()
m1 = FirstOrderLM().fit(sentences)      # first-order:  P(X_t | X_{t-1})
m2 = SecondOrderLM().fit(sentences)     # second-order: P(X_t | X_{t-2}, X_{t-1})

words = sorted({w for s in sentences for w in s})   # the 10 distinct words
V_ctx = [START] + words     # tokens that can appear in a context
V_out = words + [END]       # tokens that can be predicted
training_set = {" ".join(s) for s in sentences}     # used to detect "novel" sentences

# =============================================================================
# 2. Question 3: conditional probability tables
# =============================================================================
out = ["# First-order CPT  P(next | current)\n"]
out.append("Full table (rows = current word, columns = next word; 0 = zero-probability transition)\n")
out.append("| current \\ next | " + " | ".join(V_out) + " |")
out.append("|" + "---|" * (len(V_out) + 1))
for c in V_ctx:
    row = [f"{m1.prob(c, w):.3f}".rstrip("0").rstrip(".") if m1.prob(c, w) else "0" for w in V_out]
    out.append(f"| {c} | " + " | ".join(row) + " |")

out.append("\n## Required words: distributions and zero-probability transitions\n")
for c in ["the", "cat", "dog", "sat", "ran"]:
    zeros = [w for w in V_out if m1.prob(c, w) == 0]       # transitions never observed
    out.append(f"**P(next | {c})**: {fmt_dist(m1.distribution(c))}  ")
    out.append(f"zero-probability next words: {', '.join(zeros)}\n")
write("cpt_first_order.md", "\n".join(out))

out = ["# Second-order CPT  P(next | previous-two-words)  (observed contexts only)\n"]
out.append("| context (X_{t-2}, X_{t-1}) | distribution of X_t |")
out.append("|---|---|")
for ctx in sorted(m2.probs):
    out.append(f"| ({ctx[0]}, {ctx[1]}) | {fmt_dist(m2.distribution(ctx))} |")
write("cpt_second_order.md", "\n".join(out))

# =============================================================================
# 3. Part VII: normalisation test -- every row of every CPT must sum to 1
# =============================================================================
out = ["First-order model: sum_v P(v | w) for each context w"]
for w, t in sorted(m1.check_normalisation().items()):
    out.append(f"  {w:10s} {t:.6f}  {'OK' if abs(t-1) < 1e-9 else 'FAIL'}")
out.append("\nSecond-order model: sum_v P(v | a, b) for each observed context (a, b)")
for ctx, t in sorted(m2.check_normalisation().items()):
    out.append(f"  ({ctx[0]}, {ctx[1]})".ljust(24) + f" {t:.6f}  {'OK' if abs(t-1) < 1e-9 else 'FAIL'}")
ok = all(abs(t - 1) < 1e-9 for t in list(m1.check_normalisation().values()) + list(m2.check_normalisation().values()))
out.append(f"\nAll distributions normalised: {ok}")
write("normalisation_tests.txt", "\n".join(out))

# =============================================================================
# 4. Part VIII: predict the next word (distribution + arg max)
# =============================================================================
out = ["First-order next-word predictions\n"]
for c in [START, "the", "cat", "dog", "sat", "ran", "on", "to"]:
    out.append(f"P(X_t+1 | X_t = {c}): {fmt_dist(m1.distribution(c))}")
    out.append(f"   argmax -> {m1.most_probable(c)}\n")
out.append("Second-order next-word predictions\n")
for ctx in [(START, START), (START, "the"), ("the", "cat"), ("the", "dog"), ("on", "the"), ("to", "the"), ("cat", "ran")]:
    out.append(f"P(X_t | {ctx[0]}, {ctx[1]}): {fmt_dist(m2.distribution(ctx))}")
    out.append(f"   argmax -> {m2.most_probable(ctx)}\n")
write("predictions.txt", "\n".join(out))

# =============================================================================
# 5. Part IX: generate 20 sentences with each model (sampling)
# =============================================================================
g1 = gen_many(m1, 20, "sample", SEED)
g2 = gen_many(m2, 20, "sample", SEED)
write("generated_first_order.txt", "\n".join(sent(s) for s in g1) + "\n")
write("generated_second_order.txt", "\n".join(sent(s) for s in g2) + "\n")

# =============================================================================
# 6. Part X: Mode A (greedy) vs Mode B (sampling), 5 sentences each
# =============================================================================
out = []
for name, m in [("First-order", m1), ("Second-order", m2)]:
    out.append(f"== {name} model ==")
    out.append("Mode A (greedy, 5 sentences):")
    for s in gen_many(m, 5, "greedy", SEED):
        out.append("  " + sent(s) + ("   [hit max length, never emitted <END>]" if len(s) == MAX_LEN else ""))
    out.append("Mode B (sampling, 5 sentences):")
    for s in gen_many(m, 5, "sample", SEED):
        out.append("  " + sent(s) + ("   [hit max length]" if len(s) == MAX_LEN else ""))
    out.append("")
write("greedy_vs_sampling.txt", "\n".join(out))


# =============================================================================
# 7. Part XIII: compare first-order and second-order models
# =============================================================================
def stats(model, n=2000, seed=7):
    """Generate n sentences and measure diversity."""
    samples = gen_many(model, n, "sample", seed)
    distinct = {" ".join(s) for s in samples}                    # different sentences
    novel = {s for s in distinct if s not in training_set}       # not copied from training data
    trunc = sum(len(s) == MAX_LEN for s in samples)              # sentences cut off at max length
    return dict(distinct=len(distinct), novel=len(novel), trunc=trunc,
                mean_len=sum(len(s) for s in samples) / n, n=n,
                examples_novel=sorted(novel, key=lambda s: (len(s), s))[:6])


s1, s2 = stats(m1), stats(m2)
obs1, obs2 = len(m1.probs), len(m2.probs)              # contexts actually seen in training
poss1, poss2 = len(V_ctx), len(V_ctx) ** 2             # contexts that could exist: |V|^order
full1, full2 = poss1 * len(V_out), poss2 * len(V_out)  # full CPT size = contexts x outputs
p1, p2 = m1.num_parameters(), m2.num_parameters()      # non-zero entries actually stored

cmp = f"""Comparison of first-order and second-order models
(vocabulary: {len(words)} words; context tokens |V_ctx| = {len(V_ctx)} incl. <START>; output tokens |V_out| = {len(V_out)} incl. <END>)

| metric | first-order | second-order |
|---|---|---|
| possible contexts (|V_ctx|^n) | {poss1} | {poss2} |
| observed contexts | {obs1} | {obs2} |
| unobserved (zero-probability) contexts | {poss1-obs1} | {poss2-obs2} ({100*(poss2-obs2)/poss2:.1f}%) |
| full CPT size (contexts x |V_out|) | {full1} | {full2} |
| distinct non-zero parameters | {p1} | {p2} |
| zero entries inside the full CPT | {full1-p1} | {full2-p2} |
| distinct sentences in {s1['n']} samples | {s1['distinct']} | {s2['distinct']} |
| of which NOT in training data | {s1['novel']} | {s2['novel']} |
| samples hitting max length ({MAX_LEN}) | {s1['trunc']} | {s2['trunc']} |
| mean sentence length (words) | {s1['mean_len']:.2f} | {s2['mean_len']:.2f} |

Example novel sentences (first-order): {s1['examples_novel']}
Example novel sentences (second-order): {s2['examples_novel']}

Sentence probabilities (chain rule):
"""
for st in ["the cat sat on the mat", "the cat sat on the park", "the dog ran to the mat"]:
    cmp += (f"  P('{st}')  first-order = {2.718281828 ** m1.sentence_log_prob(st.split()):.5f}   "
            f"second-order = {2.718281828 ** m2.sentence_log_prob(st.split()):.5f}\n")
write("model_comparison.md", cmp)

# Also print a summary to the terminal.
print(open(os.path.join(OUT, "normalisation_tests.txt")).read())
print("\n" + cmp)
print(open(os.path.join(OUT, "greedy_vs_sampling.txt")).read())
print("First-order samples:\n" + "\n".join(sent(s) for s in g1))
print("\nSecond-order samples:\n" + "\n".join(sent(s) for s in g2))
