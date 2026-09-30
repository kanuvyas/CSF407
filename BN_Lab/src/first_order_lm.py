"""
first_order_lm.py -- first-order autoregressive language model (bigram model).

Bayesian network:   X1 -> X2 -> X3 -> ... -> XT
Assumption:         P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1})
Estimation:         P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)
                    where C(w_i, w_j) = how often w_j directly follows w_i.

Only ordinary Python data structures are used (dicts, Counter, random).
No machine-learning library and no pretrained model.
"""
import math
import random
from collections import Counter, defaultdict

from data import START, END


class FirstOrderLM:
    def __init__(self):
        # counts[prev][next] = C(prev, next): number of times `next` followed `prev`
        self.counts = defaultdict(Counter)
        # probs[prev][next]  = P(next | prev): the conditional probability table (CPT)
        self.probs = {}

    # ------------------------------------------------------------------
    # TRAINING: count transitions, then turn the counts into probabilities
    # ------------------------------------------------------------------
    def fit(self, sentences):
        """Build the CPT from a list of tokenised sentences."""
        self.counts = defaultdict(Counter)

        # Step 1: count every pair of consecutive tokens.
        for sent in sentences:
            tokens = [START] + list(sent) + [END]          # wrap sentence in <START> ... <END>
            for prev, nxt in zip(tokens[:-1], tokens[1:]):  # consecutive pairs
                self.counts[prev][nxt] += 1                 # C(prev, nxt) += 1

        # Step 2: normalise each row so it sums to 1  ->  P(nxt | prev)
        self.probs = {}
        for prev, next_counts in self.counts.items():
            total = sum(next_counts.values())               # denominator: sum_k C(prev, w_k)
            self.probs[prev] = {w: n / total for w, n in next_counts.items()}
        return self

    # ------------------------------------------------------------------
    # INFERENCE: read probabilities out of the CPT
    # ------------------------------------------------------------------
    def distribution(self, prev):
        """Return P(. | prev) as a dict {word: probability}, sorted alphabetically.

        Returns an empty dict if `prev` was never seen as a context.
        We use .get() so that looking up an unseen word does not crash
        (and does not silently add an empty entry, as a defaultdict would).
        """
        return dict(sorted(self.probs.get(prev, {}).items()))

    def prob(self, prev, nxt):
        """P(nxt | prev). Zero if that transition was never observed."""
        return self.probs.get(prev, {}).get(nxt, 0.0)

    def most_probable(self, prev):
        """arg max_w P(w | prev): the single most likely next word (greedy choice).

        Because distribution() is sorted alphabetically and max() returns the
        first maximal element, ties are broken alphabetically -> deterministic output.
        Returns None for an unseen context.
        """
        dist = self.distribution(prev)
        if not dist:
            return None
        return max(dist, key=dist.get)

    def sample_next(self, prev, rng):
        """Sample w ~ P(. | prev): pick a word with probability equal to its CPT entry.

        rng.choices(..., weights=...) is a weighted random draw. (random.choice
        would be WRONG here: it ignores the probabilities and picks uniformly.)
        Returns None for an unseen context.
        """
        dist = self.distribution(prev)
        if not dist:
            return None
        return rng.choices(list(dist), weights=list(dist.values()), k=1)[0]

    # ------------------------------------------------------------------
    # GENERATION: sample -> append -> sample again ... until <END>
    # ------------------------------------------------------------------
    def generate(self, mode="sample", rng=None, max_len=20):
        """Generate one sentence as a list of words (without <START>/<END>).

        mode="sample" : Mode B, sample each word from P(w | previous word)
        mode="greedy" : Mode A, always take arg max P(w | previous word)
        max_len is a safety limit: greedy decoding can loop forever
        (e.g. "the cat sat on the cat sat on ...") because <END> is never the arg max.
        """
        if rng is None:
            rng = random.Random()
        words = []
        prev = START                                        # X_1 ~ P(X_1 | <START>)
        while len(words) < max_len:
            if mode == "greedy":
                nxt = self.most_probable(prev)
            else:
                nxt = self.sample_next(prev, rng)
            if nxt is None or nxt == END:                   # stop at <END> (or unseen context)
                break
            words.append(nxt)
            prev = nxt                                      # the new word is the next context
        return words

    def sentence_log_prob(self, sentence):
        """log P(sentence), computed with the chain rule / BN factorisation.

        P(x1..xT) = P(x1|<START>) * P(x2|x1) * ... * P(<END>|xT)
        Returns -inf if any factor is zero (impossible sentence).
        """
        tokens = [START] + list(sentence) + [END]
        total = 0.0
        for prev, nxt in zip(tokens[:-1], tokens[1:]):
            p = self.prob(prev, nxt)
            if p == 0.0:
                return float("-inf")
            total += math.log(p)                            # add logs instead of multiplying
        return total

    # ------------------------------------------------------------------
    # DIAGNOSTICS: used by the tests and the model comparison
    # ------------------------------------------------------------------
    def check_normalisation(self):
        """Return sum_v P(v | w) for every context w. Every value must be ~1.0."""
        return {w: sum(d.values()) for w, d in self.probs.items()}

    def num_parameters(self):
        """Number of distinct non-zero probabilities stored in the CPT."""
        return sum(len(d) for d in self.probs.values())
