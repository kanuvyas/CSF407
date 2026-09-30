"""
second_order_lm.py -- second-order autoregressive language model (trigram model).

Bayesian network:   X_{t-2} -> X_t <- X_{t-1}    (each word has TWO parents)
Assumption:         P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-2}, X_{t-1})
Estimation:         P(c | a, b) = C(a, b, c) / sum_k C(a, b, w_k)
                    where C(a, b, c) = how often the triple (a, b, c) occurs.

The code has the same structure as first_order_lm.py; the only real change is
that the context is now a PAIR of words (a tuple) instead of a single word.
"""
import math
import random
from collections import Counter, defaultdict

from data import START, END


class SecondOrderLM:
    def __init__(self):
        # counts[(a, b)][c] = C(a, b, c): times word c followed the pair (a, b)
        self.counts = defaultdict(Counter)
        # probs[(a, b)][c]  = P(c | a, b): the second-order CPT
        self.probs = {}

    # ------------------------------------------------------------------
    # TRAINING: count observed triples, then normalise
    # ------------------------------------------------------------------
    def fit(self, sentences):
        """Build the second-order CPT from tokenised sentences.

        Each sentence is padded with TWO <START> tokens. That way the first word
        has the context (<START>, <START>) and the second word has (<START>, x1),
        so every prediction has a full two-token context and no special case is needed.
        """
        self.counts = defaultdict(Counter)

        # Step 1: count every triple of consecutive tokens (a, b, c).
        for sent in sentences:
            tokens = [START, START] + list(sent) + [END]
            for a, b, c in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
                self.counts[(a, b)][c] += 1                 # C(a, b, c) += 1

        # Step 2: normalise each context row so it sums to 1.
        self.probs = {}
        for ctx, next_counts in self.counts.items():
            total = sum(next_counts.values())               # sum_k C(a, b, w_k)
            self.probs[ctx] = {w: n / total for w, n in next_counts.items()}
        return self

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------
    def distribution(self, ctx):
        """P(. | ctx) for a context pair ctx = (a, b); {} if the pair was never seen."""
        return dict(sorted(self.probs.get(tuple(ctx), {}).items()))

    def prob(self, ctx, nxt):
        """P(nxt | ctx); zero if that triple was never observed."""
        return self.probs.get(tuple(ctx), {}).get(nxt, 0.0)

    def most_probable(self, ctx):
        """arg max_w P(w | ctx); ties broken alphabetically; None if ctx unseen."""
        dist = self.distribution(ctx)
        if not dist:
            return None
        return max(dist, key=dist.get)

    def sample_next(self, ctx, rng):
        """Sample w ~ P(. | ctx) using a weighted random draw; None if ctx unseen."""
        dist = self.distribution(ctx)
        if not dist:
            return None
        return rng.choices(list(dist), weights=list(dist.values()), k=1)[0]

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------
    def generate(self, mode="sample", rng=None, max_len=20):
        """Generate one sentence (list of words). Same modes as the first-order model."""
        if rng is None:
            rng = random.Random()
        words = []
        ctx = (START, START)                                # start with an empty history
        while len(words) < max_len:
            if mode == "greedy":
                nxt = self.most_probable(ctx)
            else:
                nxt = self.sample_next(ctx, rng)
            if nxt is None or nxt == END:
                break
            words.append(nxt)
            ctx = (ctx[1], nxt)                             # slide the 2-word window forward
        return words

    def sentence_log_prob(self, sentence):
        """log P(sentence) with the second-order factorisation (-inf if impossible)."""
        tokens = [START, START] + list(sentence) + [END]
        total = 0.0
        for a, b, c in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
            p = self.prob((a, b), c)
            if p == 0.0:
                return float("-inf")
            total += math.log(p)
        return total

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------
    def check_normalisation(self):
        """sum_v P(v | a, b) for every observed context; each should be ~1.0."""
        return {ctx: sum(d.values()) for ctx, d in self.probs.items()}

    def num_parameters(self):
        """Number of distinct non-zero probabilities stored in the CPT."""
        return sum(len(d) for d in self.probs.values())
