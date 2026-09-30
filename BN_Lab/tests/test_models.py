"""
test_models.py -- property tests for the first-order and second-order language models.

The handout asks us to test a probabilistic program by checking properties that MUST
hold if the code really implements the intended model, instead of eyeballing output.
The tests are grouped by the property they protect:

  A. Data and tokens        the input is what the handout says it is
  B. Counting               the count tables equal hand counts
  C. Normalisation          every row of every CPT is a distribution (Part VII)
  D. Prediction             arg max, ties, unseen words (Parts VIII, Q7)
  E. Sampling               weighted sampling reproduces the CPT (Parts IX, X)
  F. Chain rule             sentence probabilities multiply out, and total mass = 1
  G. First vs second order  the second-order model is a refinement of the first
  H. Bug detection          deliberately broken models must FAIL our checks

Run:   python tests/test_models.py          (plain runner, prints a summary)
  or:  python -m unittest discover -s tests -v
  or:  python -m pytest tests
"""
import math
import os
import random
import sys
import unittest
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import START, END, load_sentences
from first_order_lm import FirstOrderLM
from second_order_lm import SecondOrderLM

SENTENCES = load_sentences()
M1 = FirstOrderLM().fit(SENTENCES)
M2 = SecondOrderLM().fit(SENTENCES)


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def total_variation(freq, dist, n):
    """Total-variation distance between empirical frequencies and a distribution."""
    words = set(freq) | set(dist)
    return 0.5 * sum(abs(freq.get(w, 0) / n - dist.get(w, 0.0)) for w in words)


def terminated_mass(model, start_state, advance, max_steps):
    """Exact probability that the model emits <END> within `max_steps` words.

    Walks the Markov chain forward one step at a time, keeping the probability of
    each live state, and adds up the probability that flows into <END>.
    No sampling is involved, so the answer is exact.
    """
    live = {start_state: 1.0}
    ended = 0.0
    for _ in range(max_steps):
        nxt_live = {}
        for state, p in live.items():
            for word, q in model.distribution(state).items():
                if word == END:
                    ended += p * q
                else:
                    s2 = advance(state, word)
                    nxt_live[s2] = nxt_live.get(s2, 0.0) + p * q
        live = nxt_live
    return ended


# =============================================================================
# A. Data and tokens
# =============================================================================
class TestData(unittest.TestCase):
    def test_six_sentences_all_lowercase(self):
        self.assertEqual(len(SENTENCES), 6)
        for s in SENTENCES:
            for w in s:
                self.assertEqual(w, w.lower())

    def test_special_tokens_not_in_raw_data(self):
        # <START>/<END> are added by the models, never present in the plain sentences.
        for s in SENTENCES:
            self.assertNotIn(START, s)
            self.assertNotIn(END, s)

    def test_vocabulary_is_ten_words(self):
        vocab = {w for s in SENTENCES for w in s}
        self.assertEqual(vocab, {"the", "cat", "dog", "sat", "ran", "on", "to",
                                 "mat", "rug", "park"})

    def test_uppercase_input_is_lowercased(self):
        # load_sentences() must lower-case whatever text it is given.
        import data
        original = data.RAW_TEXT
        try:
            data.RAW_TEXT = "The CAT Sat ON the Mat\nTHE DOG RAN"
            self.assertEqual(data.load_sentences(),
                             [["the", "cat", "sat", "on", "the", "mat"], ["the", "dog", "ran"]])
        finally:
            data.RAW_TEXT = original


# =============================================================================
# B. Counting
# =============================================================================
class TestCounts(unittest.TestCase):
    def test_first_order_total_transitions(self):
        # each sentence has 6 words -> 7 transitions including <START> and <END>
        total = sum(sum(c.values()) for c in M1.counts.values())
        self.assertEqual(total, 6 * 7)

    def test_second_order_total_triples(self):
        total = sum(sum(c.values()) for c in M2.counts.values())
        self.assertEqual(total, 6 * 7)

    def test_selected_first_order_counts(self):
        self.assertEqual(M1.counts["the"]["cat"], 3)
        self.assertEqual(M1.counts["the"]["dog"], 3)
        self.assertEqual(M1.counts["the"]["mat"], 2)
        self.assertEqual(M1.counts["sat"]["on"], 4)
        self.assertEqual(sum(M1.counts["the"].values()), 12)   # 'the' occurs 12 times

    def test_handout_example_three_cat_two_dog(self):
        # Handout Part IV: 'the cat' three times and 'the dog' twice -> 3/5 and 2/5.
        toy = FirstOrderLM().fit([["the", "cat"]] * 3 + [["the", "dog"]] * 2)
        self.assertTrue(close(toy.prob("the", "cat"), 3 / 5))
        self.assertTrue(close(toy.prob("the", "dog"), 2 / 5))

    def test_counts_independent_of_sentence_order(self):
        shuffled = SENTENCES[::-1]
        other = FirstOrderLM().fit(shuffled)
        self.assertEqual({k: dict(v) for k, v in other.counts.items()},
                         {k: dict(v) for k, v in M1.counts.items()})

    def test_refit_replaces_old_counts(self):
        # Calling fit() twice must not accumulate counts from the first call.
        m = FirstOrderLM().fit(SENTENCES).fit(SENTENCES)
        self.assertEqual(m.counts["the"]["cat"], 3)


# =============================================================================
# C. Normalisation (Part VII)
# =============================================================================
class TestNormalisation(unittest.TestCase):
    def test_first_order_rows_sum_to_one(self):
        for word, total in M1.check_normalisation().items():
            self.assertTrue(close(total, 1.0), (word, total))

    def test_second_order_rows_sum_to_one(self):
        for ctx, total in M2.check_normalisation().items():
            self.assertTrue(close(total, 1.0), (ctx, total))

    def test_every_probability_in_unit_interval(self):
        for model in (M1, M2):
            for ctx in model.probs:
                for p in model.distribution(ctx).values():
                    self.assertGreater(p, 0.0)      # stored entries are non-zero
                    self.assertLessEqual(p, 1.0)

    def test_number_of_rows(self):
        self.assertEqual(len(M1.probs), 11)   # <START> + 10 words
        self.assertEqual(len(M2.probs), 15)   # observed word pairs

    def test_end_is_never_a_context(self):
        # nothing follows <END>, so it must not have a row
        self.assertNotIn(END, M1.probs)
        self.assertFalse(any(ctx[0] == END for ctx in M2.probs))


# =============================================================================
# D. Prediction (Part VIII, Q7)
# =============================================================================
class TestPrediction(unittest.TestCase):
    def test_argmax_first_order(self):
        expected = {"cat": "sat", "dog": "sat", "sat": "on", "ran": "to",
                    "on": "the", "to": "the", "mat": END, START: "the"}
        for ctx, word in expected.items():
            self.assertEqual(M1.most_probable(ctx), word, ctx)

    def test_tie_is_broken_alphabetically(self):
        # P(cat|the) == P(dog|the) == 0.25 -> 'cat' wins because it is first alphabetically
        self.assertTrue(close(M1.prob("the", "cat"), M1.prob("the", "dog")))
        self.assertEqual(M1.most_probable("the"), "cat")
        # (<START>, the): cat and dog both 0.5
        self.assertEqual(M2.most_probable((START, "the")), "cat")

    def test_more_context_changes_the_prediction(self):
        # First order: after 'the' the best word is 'cat' (0.25).
        # Second order after (to, the) the only possible word is 'park'.
        self.assertEqual(M1.most_probable("the"), "cat")
        self.assertEqual(M2.most_probable(("to", "the")), "park")

    def test_zero_probability_transitions(self):
        for prev, nxt in [("cat", "dog"), ("sat", "the"), ("ran", "on"), ("the", "sat")]:
            self.assertEqual(M1.prob(prev, nxt), 0.0)
        self.assertEqual(M2.prob(("on", "the"), "park"), 0.0)

    def test_unseen_context_does_not_crash(self):
        self.assertEqual(M1.distribution("banana"), {})
        self.assertIsNone(M1.most_probable("banana"))
        self.assertIsNone(M1.sample_next("banana", random.Random(0)))
        self.assertEqual(M1.prob("banana", "the"), 0.0)
        self.assertEqual(M2.distribution(("banana", "the")), {})
        self.assertIsNone(M2.most_probable(("the", "the")))

    def test_lookup_does_not_add_rows(self):
        # A defaultdict lookup would silently create an empty row; ours must not.
        before = (len(M1.probs), len(M1.counts), len(M2.probs), len(M2.counts))
        M1.distribution("banana"); M1.prob("banana", "x"); M2.distribution(("a", "b"))
        after = (len(M1.probs), len(M1.counts), len(M2.probs), len(M2.counts))
        self.assertEqual(before, after)

    def test_unseen_word_in_sentence_gives_minus_infinity(self):
        self.assertEqual(M1.sentence_log_prob(["the", "banana"]), float("-inf"))
        self.assertEqual(M2.sentence_log_prob(["the", "banana"]), float("-inf"))


# =============================================================================
# E. Sampling and generation (Parts IX, X)
# =============================================================================
class TestSampling(unittest.TestCase):
    N = 30000

    def test_first_order_sampling_matches_cpt_for_several_contexts(self):
        rng = random.Random(123)
        for ctx in ["the", "cat", "dog", START]:
            freq = Counter(M1.sample_next(ctx, rng) for _ in range(self.N))
            tv = total_variation(freq, M1.distribution(ctx), self.N)
            self.assertLess(tv, 0.01, (ctx, tv))

    def test_second_order_sampling_matches_cpt(self):
        rng = random.Random(321)
        for ctx in [(START, "the"), ("the", "cat"), ("on", "the")]:
            freq = Counter(M2.sample_next(ctx, rng) for _ in range(self.N))
            tv = total_variation(freq, M2.distribution(ctx), self.N)
            self.assertLess(tv, 0.01, (ctx, tv))

    def test_sampling_never_returns_a_zero_probability_word(self):
        rng = random.Random(5)
        for _ in range(3000):
            self.assertGreater(M1.prob("the", M1.sample_next("the", rng)), 0.0)

    def test_same_seed_same_sentences(self):
        a = [M1.generate("sample", random.Random(9)) for _ in range(1)]
        b = [M1.generate("sample", random.Random(9)) for _ in range(1)]
        self.assertEqual(a, b)

    def test_different_seeds_give_variety(self):
        sents = {tuple(M1.generate("sample", random.Random(s))) for s in range(200)}
        self.assertGreater(len(sents), 20)

    def test_greedy_is_deterministic_and_ignores_rng(self):
        for m in (M1, M2):
            a = m.generate("greedy", random.Random(1))
            b = m.generate("greedy", random.Random(999))
            self.assertEqual(a, b)

    def test_greedy_outputs(self):
        # first-order greedy loops: the cat sat on the cat sat on ... (cut at max_len)
        g1 = M1.generate("greedy")
        self.assertEqual(len(g1), 20)
        self.assertEqual(g1[:6], ["the", "cat", "sat", "on", "the", "cat"])
        # second-order greedy ends properly
        self.assertEqual(M2.generate("greedy"), "the cat sat on the mat".split())

    def test_max_len_is_respected(self):
        for n in (1, 5, 13):
            self.assertLessEqual(len(M1.generate("greedy", max_len=n)), n)
            self.assertLessEqual(len(M1.generate("sample", random.Random(0), max_len=n)), n)

    def test_generated_text_has_no_special_tokens(self):
        rng = random.Random(4)
        for _ in range(300):
            for m in (M1, M2):
                s = m.generate("sample", rng)
                self.assertNotIn(START, s)
                self.assertNotIn(END, s)

    def test_second_order_only_reproduces_training_sentences(self):
        training = {tuple(s) for s in SENTENCES}
        rng = random.Random(77)
        for _ in range(1000):
            self.assertIn(tuple(M2.generate("sample", rng)), training)

    def test_first_order_invents_new_sentences(self):
        training = {tuple(s) for s in SENTENCES}
        rng = random.Random(77)
        novel = [s for s in (tuple(M1.generate("sample", rng)) for _ in range(1000))
                 if s not in training]
        self.assertGreater(len(novel), 500)


# =============================================================================
# F. Chain rule and total probability mass
# =============================================================================
class TestChainRule(unittest.TestCase):
    def test_sentence_probability_first_order_by_hand(self):
        # P(the cat sat on the mat) = 1 * 3/12 * 2/3 * 1 * 1 * 2/12 * 1 = 1/36
        p = math.exp(M1.sentence_log_prob("the cat sat on the mat".split()))
        self.assertAlmostEqual(p, 1 / 36, places=12)

    def test_sentence_probability_second_order_by_hand(self):
        # P = 1 * 1/2 * 2/3 * 1 * 1 * 1/2 * 1 = 1/6
        p = math.exp(M2.sentence_log_prob("the cat sat on the mat".split()))
        self.assertAlmostEqual(p, 1 / 6, places=12)

    def test_second_order_gives_zero_to_unseen_combinations(self):
        self.assertEqual(M2.sentence_log_prob("the cat sat on the park".split()), float("-inf"))
        self.assertGreater(M1.sentence_log_prob("the cat sat on the park".split()), float("-inf"))

    def test_second_order_total_mass_is_exactly_one(self):
        # The second-order model can only produce the 6 training sentences, so the
        # probabilities of ALL sentences it can generate must add up to 1.
        mass = terminated_mass(M2, (START, START), lambda s, w: (s[1], w), 12)
        self.assertAlmostEqual(mass, 1.0, places=12)

    def test_first_order_total_mass_approaches_one_from_below(self):
        # Sentences can be arbitrarily long (loops), so the mass of sentences that
        # ended within L words increases with L, never exceeds 1, and tends to 1.
        prev = 0.0
        for L in (5, 10, 20, 40, 80):
            m = terminated_mass(M1, START, lambda s, w: w, L)
            self.assertGreater(m, prev)
            self.assertLessEqual(m, 1.0 + 1e-12)
            prev = m
        self.assertGreater(prev, 0.999)

    def test_sentence_frequencies_follow_the_chain_rule(self):
        # 'the park' = P(the|<START>) * P(park|the) * P(<END>|park) = 2/12
        rng = random.Random(2024)
        n = 30000
        hits = sum(M1.generate("sample", rng) == ["the", "park"] for _ in range(n))
        self.assertAlmostEqual(hits / n, 2 / 12, delta=0.01)


# =============================================================================
# G. Second order is a refinement of first order
# =============================================================================
class TestModelRelationship(unittest.TestCase):
    def test_pair_counts_agree(self):
        # Summing the triple counts over the first word must give the pair counts:
        # C1(b, c) = sum_a C2(a, b, c).
        collapsed = Counter()
        for (a, b), row in M2.counts.items():
            for c, n in row.items():
                collapsed[(b, c)] += n
        for b, row in M1.counts.items():
            for c, n in row.items():
                self.assertEqual(collapsed[(b, c)], n, (b, c))
        self.assertEqual(sum(collapsed.values()), sum(sum(r.values()) for r in M1.counts.values()))

    def test_second_order_support_is_inside_first_order_support(self):
        # If the longer context allows a word, the shorter context must allow it too.
        for (a, b), row in M2.probs.items():
            for c in row:
                self.assertGreater(M1.prob(b, c), 0.0, (a, b, c))

    def test_parameter_counts(self):
        self.assertEqual(M1.num_parameters(), 17)
        self.assertEqual(M2.num_parameters(), 19)

    def test_unobserved_contexts(self):
        contexts = len(M1.probs) ** 2                     # |V_ctx|^2 = 121
        self.assertEqual(contexts - len(M2.probs), 106)


# =============================================================================
# H. Bug detection: the tests must FAIL on broken models
# =============================================================================
class WrongDenominator(FirstOrderLM):
    """Divides by (row total + 1). Rows sum to less than 1."""
    def fit(self, sentences):
        super().fit(sentences)
        self.probs = {p: {w: n / (sum(c.values()) + 1) for w, n in c.items()}
                      for p, c in self.counts.items()}
        return self


class SwallowedEnd(FirstOrderLM):
    """Forgets to count transitions into <END>, but keeps the old row totals."""
    def fit(self, sentences):
        super().fit(sentences)
        self.probs = {p: {w: n / sum(c.values()) for w, n in c.items() if w != END}
                      for p, c in self.counts.items()}
        return self


class UniformSampler(FirstOrderLM):
    """Uses uniform random.choice instead of a weighted draw (ignores the CPT)."""
    def sample_next(self, prev, rng):
        dist = self.distribution(prev)
        return rng.choice(list(dist)) if dist else None


class OneStartPad(SecondOrderLM):
    """Second-order model padded with a single <START>: the first word has no full context."""
    def fit(self, sentences):
        self.counts = defaultdict(Counter)
        for sent in sentences:
            tokens = [START] + list(sent) + [END]
            for a, b, c in zip(tokens[:-2], tokens[1:-1], tokens[2:]):
                self.counts[(a, b)][c] += 1
        self.probs = {ctx: {w: n / sum(r.values()) for w, n in r.items()}
                      for ctx, r in self.counts.items()}
        return self


def rows_all_normalised(model):
    return all(close(t, 1.0) for t in model.check_normalisation().values())


def sampling_matches(model, ctx="the", n=30000):
    rng = random.Random(0)
    freq = Counter(model.sample_next(ctx, rng) for _ in range(n))
    return total_variation(freq, model.distribution(ctx), n) < 0.01


class TestBugDetection(unittest.TestCase):
    def test_correct_models_pass_both_checks(self):
        self.assertTrue(rows_all_normalised(M1))
        self.assertTrue(sampling_matches(M1))

    def test_wrong_denominator_is_caught(self):
        bad = WrongDenominator().fit(SENTENCES)
        self.assertFalse(rows_all_normalised(bad))
        self.assertTrue(close(bad.check_normalisation()[START], 6 / 7))   # 0.857

    def test_wrong_denominator_is_invisible_to_sampling(self):
        # random.choices rescales weights, so sampling looks fine even though the CPT
        # is wrong: only the normalisation test finds this bug.
        bad = WrongDenominator().fit(SENTENCES)
        rng = random.Random(0)
        n = 30000
        freq = Counter(bad.sample_next("the", rng) for _ in range(n))
        # sampled frequencies still match the CORRECT distribution ...
        self.assertLess(total_variation(freq, M1.distribution("the"), n), 0.01)
        # ... yet the stored table is wrong, and only the sum-to-1 check sees it.
        self.assertFalse(rows_all_normalised(bad))

    def test_missing_end_transitions_are_caught(self):
        bad = SwallowedEnd().fit(SENTENCES)
        self.assertFalse(rows_all_normalised(bad))
        self.assertLess(bad.check_normalisation()["mat"], 1.0)   # 'mat' row loses everything

    def test_uniform_sampler_is_caught_by_frequency_test(self):
        bad = UniformSampler().fit(SENTENCES)
        self.assertTrue(rows_all_normalised(bad))       # the CPT itself is fine ...
        self.assertFalse(sampling_matches(bad))         # ... but sampling ignores it

    def test_single_start_pad_is_caught_by_hand_values(self):
        bad = OneStartPad().fit(SENTENCES)
        self.assertEqual(bad.prob((START, START), "the"), 0.0)   # correct model gives 1.0
        self.assertEqual(M2.prob((START, START), "the"), 1.0)


# =============================================================================
if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
