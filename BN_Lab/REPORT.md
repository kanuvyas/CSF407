# Lab Report: Bayesian Networks and Autoregressive Language Models

**Course:** CS F407, AI Laboratory
**Data:** the six handout sentences, lower-cased, one word = one token, wrapped in `<START> ... <END>`
**Reproduce:** `python src/run_lab.py` (seed 42) rebuilds `results/`; `python tests/test_models.py` runs the 49 tests.
All numbers below come from those two commands. The vocabulary has 10 words, so a context can be one of 11 tokens (10 words + `<START>`) and a prediction one of 11 tokens (10 words + `<END>`).

## Contents

| Handout part | Questions | Section |
|---|---|---|
| I-II | Q1, Q2 | 1 |
| III-IV | Q3 | 2 |
| V-VI | Q4-Q7 | 3 |
| VII | Q8 | 4 |
| VIII | Q9 | 5 |
| IX-X | Q10 | 6 |
| XI-XIII | Q11, Q12 | 7 |
| XIV | (none) | 8 |
| XV | Q13 | 9 |
| Final | Q14 | 10 |
| Deliverable 7 | LLM reflection | 11 |

---

## 1. From probability to language (Parts I-II)

**Q1. Why is the chain-rule decomposition useful for generating text?**

Take "the cat sat on the mat". The chain rule writes its probability as

`P(X1..X6) = P(the) P(cat | the) P(sat | the, cat) P(on | the, cat, sat) P(the | ...) P(mat | ...)`.

The useful part is that it turns one impossible task into six small ones. A table over all whole sentences is unmanageable, but "which word comes next, given what came before?" is a small distribution over the vocabulary. We can estimate it from data and sample from it. Generation is then a loop: draw a word, append it, draw the next one, and stop when `<END>` is drawn. The decomposition is exact (it is just the definition of conditional probability), so no approximation is made until we choose what to condition on. It also handles sentences of any length, because length is decided by when `<END>` appears.

**Q2. What independence assumption does `X1 -> X2 -> X3 -> X4` make?**

```
X1 --> X2 --> X3 --> X4          P(X1,X2,X3,X4) = P(X1) P(X2|X1) P(X3|X2) P(X4|X3)
```

Every word depends only on the word immediately before it:

`P(X_t | X_1, ..., X_{t-1}) = P(X_t | X_{t-1})`, equivalently `X_t ⊥ {X_1, ..., X_{t-2}} | X_{t-1}`.

In graph terms, `X_{t-1}` blocks the path from all earlier words to `X_t`. The price is that the model forgets everything older than one word. After "the cat sat on the" it only sees "the", so "... on the cat sat on ..." looks perfectly normal to it (see Q10).

---

## 2. The dataset and the conditional probability table (Parts III-IV)

The six sentences are converted to lower case and split into words. `<START>` and `<END>` are added inside `fit()`, so each sentence gives 7 transitions and the data contains 6 x 7 = 42 transitions in total. The estimator is

`P(w_j | w_i) = C(w_i, w_j) / sum_k C(w_i, w_k)`.

As a check on my understanding of the handout's example: if "the cat" occurred 3 times and "the dog" twice, the counts give 3/5 and 2/5. The same rule applied to our data gives, for example, `C(the, cat) = 3` out of `C(the, *) = 12`.

**Q3. CPT `P(next | current)` for the required words** (`results/cpt_first_order.md` has the full 11 x 11 table)

| current | non-zero next words | transitions with probability 0 |
|---|---|---|
| the | cat 3/12 = 0.250, dog 3/12 = 0.250, mat 2/12 = 0.167, park 2/12 = 0.167, rug 2/12 = 0.167 | the, sat, ran, on, to, `<END>` |
| cat | sat 2/3 = 0.667, ran 1/3 = 0.333 | cat, dog, mat, on, park, rug, the, to, `<END>` |
| dog | sat 2/3 = 0.667, ran 1/3 = 0.333 | same nine as `cat` |
| sat | on 4/4 = 1.000 | all other 10 |
| ran | to 2/2 = 1.000 | all other 10 |

Zero-probability transitions come in two kinds:
- *Sensible zeros*, which the data got right: `sat -> the`, `the -> sat`, `cat -> cat`.
- *Accidental zeros*, caused by having only six sentences: "the cat ran on the mat" is good English, yet `ran -> on` gets probability 0 because it never occurred in the data.

Over the whole table only 17 of 121 cells are non-zero (104 are zero). The model cannot tell "impossible" from "not seen yet".

---

## 3. Asking the LLM and inspecting the code (Parts V-VI)

The exact specifications I gave are in `prompts.md` (Prompt 1 for the first-order model, Prompt 2 for the two modes). Before prompting I wrote down the variables, the dependency structure, the estimator and what the program had to compute, as Section 3 of the handout asks. Section 11 below shows what I checked in the returned code.

**Q4. Where are the transition counts stored?**
In `self.counts`, a `defaultdict(Counter)` in `src/first_order_lm.py`. `self.counts[prev][nxt]` holds `C(prev, nxt)`. It is filled in `fit()` at line 37 (`self.counts[prev][nxt] += 1`).

**Q5. Where is `P(X_t | X_{t-1})` computed?**
At the end of `fit()`. Line 42 sums a row (`total = sum(next_counts.values())`) and line 43 divides each count by it (`n / total`). The results are stored in `self.probs[prev][next]`, and that dictionary is the CPT. `distribution()` and `prob()` only read from it.

**Q6. How does the program choose the next word? Is it arg max or sampling? What is the difference?**
Both, selected by the `mode` argument of `generate()`.
- `mode="greedy"` calls `most_probable()`, which returns `max(dist, key=dist.get)` (line 72). It always picks the single most likely word, so the output is fixed.
- `mode="sample"` calls `sample_next()`, which draws with `rng.choices(..., weights=probabilities)` (line 84). Each word is chosen with probability equal to its CPT entry: after "the", `cat` 25% of the time, `park` 16.7%, and so on.

The difference matters. Greedy throws away everything except the top entry, so it has no variety and is easily trapped in loops. Sampling reproduces the whole distribution the model learned. Note that `random.choice` would be wrong in `sample_next`, since it ignores the weights and picks uniformly; the test suite has a deliberately broken `UniformSampler` to show that this bug is detected.

**Q7. What happens for a word with no observed transition?**
The lookup uses `.get(prev, {})`, so nothing crashes: `distribution()` returns `{}`, `prob()` returns 0.0, `most_probable()` and `sample_next()` return `None`, and `generate()` stops. Using `.get` also means the lookup does not silently create an empty row, which a plain `defaultdict` lookup would (the tests check this). There is no smoothing, so the model has nothing to say about a word it has never seen. A sentence containing such a word gets log-probability `-inf`.

---

## 4. Testing the probability model (Part VII)

For every context `w`, `sum_v P(v | w)` must be 1. `check_normalisation()` returns this sum for every row, and `run_lab.py` prints it (`results/normalisation_tests.txt`):

| Model | Rows checked | Rows summing to 1.000000 |
|---|---|---|
| first-order | 11 | 11 |
| second-order | 15 | 15 |

The test suite (`tests/test_models.py`, log in `results/test_output.txt`) goes further. It checks counts against hand counts, that sampled frequencies match the CPT, that sentence probabilities equal hand-computed chain-rule products, and that broken models fail.

**Q8. If one of the totals is 0.87, what does it tell you?**
The implementation is wrong. That row is not a probability distribution, and 13% of the probability mass is missing. Typical causes are a wrong denominator, a next word dropped from the row while still counted in the total, or smoothing applied to only some entries. I reproduced the first cause with `WrongDenominator`, which divides by `total + 1`: the `<START>` row then sums to 6/7 = 0.857, and the test detects it.

The dangerous part is that the bug is invisible in the output. `rng.choices` rescales its weights internally, so a broken row still produces sensible-looking sentences with the correct proportions. `test_wrong_denominator_is_invisible_to_sampling` demonstrates this: the frequencies are right, the table is wrong, and only the sum-to-1 test notices.

---

## 5. Predicting the next word (Part VIII)

`results/predictions.txt` lists eight first-order contexts. The table shows the ones I discuss.

| context | P(next \| context) | arg max | what I would expect |
|---|---|---|---|
| `<START>` | the 1.0 | the | the |
| the | cat .25, dog .25, mat/park/rug .167 | cat (tie with dog) | a noun, but which one? |
| cat | sat .667, ran .333 | sat | sat / ran |
| dog | sat .667, ran .333 | sat | sat / ran |
| sat | on 1.0 | on | on |
| ran | to 1.0 | to | to |
| on | the 1.0 | the | the |
| to | the 1.0 | the | the |

**Q9. Are the most probable predictions always the ones I would expect?**
No, and the exceptions are informative.
- Where the data is regular (`sat -> on`, `ran -> to`, `on -> the`) the model matches intuition.
- After "the", `cat` wins only because of a tie with `dog` (0.25 each), broken alphabetically. This is a property of the code, not of language, and it says nothing about cats being more natural.
- Mixed contexts fail: after "sat on the" a human knows the next word is a surface (`mat`, `rug`), not `cat`, but the first-order model only sees "the" and gives `cat` and `dog` the highest weight.

A probability model reports how often things followed each other in *its data*. A human's expectation comes from grammar, meaning and world knowledge (mats are things you sit on). The model has none of these, so it is right exactly as far as the sample of six sentences is representative.

---

## 6. Generating text and comparing the two modes (Parts IX-X)

**Part IX.** `run_lab.py` samples 20 sentences from each model (`results/generated_first_order.txt`, `results/generated_second_order.txt`). Summary of those 20:

| | first-order | second-order |
|---|---|---|
| exact copies of a training sentence | 4 | 20 |
| distinct sentences | 16 | 6 |
| two-word "sentences" (e.g. `the mat`) | 7 | 0 |
| cut off at 20 words | 1 | 0 |

**Part X.** `results/greedy_vs_sampling.txt` has five sentences from each mode for each model. First-order, Mode A gives the identical 20-word line five times (`the cat sat on the cat sat on the cat sat on ...`); Mode B gives five different sentences, e.g. `the cat ran to the park` and `the dog ran to the cat ran to the cat sat on the park`. Second-order, Mode A gives `the cat sat on the mat` five times; Mode B gives three different sentences in five draws.

**Q10. Which mode produces more variation, and why?**
Sampling. Greedy has no randomness at all: at every step it takes the arg max, so the whole path is fixed by the CPT and every run is identical. Sampling makes a fresh random choice at each position, and different choices lead to different sentences. Over 2000 samples the first-order model produced 292 distinct sentences and the second-order model only 6.

Greedy also has a failure mode. In the first-order model `<END>` is never the arg max anywhere on the path `the -> cat -> sat -> on -> the`, so the path becomes a 4-word cycle and the sentence is only ended by the `max_len = 20` safety limit. Sampling escapes such loops because `<END>` has non-zero probability after `mat`, `rug` and `park` (71 of 2000 first-order samples still hit the limit).

---

## 7. A second-order Bayesian network (Parts XI-XIII)

`results/model_comparison.md` (2000 samples per model, seed 7):

| metric | first-order | second-order |
|---|---|---|
| possible contexts (`11^n`) | 11 | 121 |
| contexts seen in training | 11 | 15 |
| unobserved (zero-probability) contexts | 0 | 106 (87.6%) |
| full CPT size (contexts x 11 outputs) | 121 | 1331 |
| distinct non-zero parameters | 17 | 19 |
| non-zero share of the full CPT | 14.0% | 1.4% |
| distinct sentences in 2000 samples | 292 | 6 |
| of those, not in the training data | 286 | 0 |
| samples cut off at 20 words | 71 | 0 |

**Q11. How does the second-order model differ from the first-order one?**
1. *Graph structure.* Each word has two parents instead of one: `X_{t-2} -> X_t <- X_{t-1}`. The four-word factorisation becomes `P(X1) P(X2|X1) P(X3|X1,X2) P(X4|X2,X3)`.
2. *CPT.* Rows are indexed by word *pairs*. There are up to 11^2 = 121 possible rows (against 11), of which 15 are observed; the full table has 1331 cells (against 121).
3. *Context.* Two previous words. This is what fixes "on the -> mat / rug only".
4. *Data needed.* Far more. The table is 11 times bigger, but there are still only 42 observed transitions. Only 19 cells are non-zero.

**Q12. Why does more context help, and why does it make estimation harder?**
More context sharpens the prediction because different histories can get different distributions. After `(on, the)` the second-order model gives `mat` 0.5, `rug` 0.5 and `park` 0; the first-order model, seeing just "the", allows `park` with probability 0.167 and so accepts "the cat sat on the park". That is why the second-order model assigns this sentence probability 0 while the first-order model gives it 0.0278.

The cost is the size of the CPT. A model with context length n has `|V|^n` rows, so the table grows exponentially in n (11, then 121, then 1331 rows for n = 1, 2, 3 in our setting). Each row must be estimated from the few times its context occurred, and most contexts never occur: 106 of the 121 second-order rows are empty. The estimate for a seen context is a frequency from very few samples, so the model ends up *memorising* the training set. It generated 0 new sentences in 2000 samples.

**Qualitative coherence (Part XIII asks for examples, not just numbers)**

| | first-order | second-order |
|---|---|---|
| Grammatical, new | `the cat ran to the mat`, `the dog ran to the mat` | none |
| Fragments | `the mat`, `the park`, `the rug` (7 of the 20 samples) | none |
| Wrong across a longer range | `the cat sat on the cat sat on the park` | none |
| Exact copies of training | 4 of 20 | 20 of 20 |

The first-order model is creative but sloppy; the second-order model is fluent but only repeats. Two sentences show the trade-off directly: `P("the cat sat on the mat")` is 0.0278 (first-order) and 0.1667 (second-order), while `P("the dog ran to the mat")` is 0.0139 against exactly 0. With this little data, neither model is ideal.

---

## 8. Connection to modern language models (Part XIV)

A modern autoregressive LM also models `P(X_t | X_1, ..., X_{t-1})`, and the joint probability is again the product of these conditionals, with `P(x_1 | <START>)` as the first factor. That is exactly how our first factor `P(the | <START>)` is stored. What changes is how the conditional is *represented*: a neural network takes the whole history and outputs a distribution, instead of a lookup table over the last one or two words. The table in the handout summarises the differences (CPTs vs network weights, fixed window vs large learned context, counting vs gradient training). Generation is the same in both: sample, append, repeat. The n-gram models here are not comparable in power to a large language model, but the probabilistic question is the same.

---

## 9. Approach A versus Approach B (Part XV)

**Q13. Why is Approach B ("implement P(X_t | X_{t-1}) from transition counts, with sampling") better?**

- *Specifying intended behaviour.* Approach A leaves the important decisions to the LLM: word-level or character-level? counts or a neural network? sampling or arg max? Approach B fixes the variables, the estimator and the generation method, so a correct answer is defined.
- *Understanding the representation.* Because I knew the program had to contain a count table and a row-normalised probability table, I knew what to look for when reading the code (Q4, Q5). With Approach A I would not know what I was reading.
- *Validating the implementation.* B gives a specification to check the code against. Under A, code that "runs and prints sentences" is the only test available.
- *Testing probabilistic invariants.* Rows must sum to 1, the entries must match hand counts, and sample frequencies must match the CPT. These properties follow from the model, not from the code. This matters because bugs like a wrong denominator do not change how the output looks (Q8).
- *Implementation versus model.* The model is the mathematical object `P(X_t | X_{t-1})`; the Python is one implementation of it. Keeping them separate means the implementation could be replaced (for example by a neural network) while the same invariants are still tested. Without a stated model there is nothing to distinguish a bug from a design choice.

---

## 10. What did the Bayesian network add? (Final question)

**Q14.** Five things, each tied to something concrete in this lab:

1. *A representation of dependencies.* The picture `X_{t-1} -> X_t` (or with two parents) says at a glance which variables each word depends on, before any code is written.
2. *A factorisation of the joint distribution.* The graph gives the formula directly: `P(X1..X4) = P(X1) P(X2|X1) P(X3|X2) P(X4|X3)`. It tells us which tables to build and lets us score whole sentences, for example `P("the cat sat on the mat") = 1 x 3/12 x 2/3 x 1 x 1 x 2/12 x 1 = 1/36` (first order), checked in the tests.
3. *Reasoning about independence assumptions.* A missing edge is an assumption. The absence of an edge from `X_{t-2}` to `X_t` is precisely why the first-order model cannot see that "the cat sat on the" should be followed by a surface.
4. *The effect of increasing context.* Adding one edge changes the CPT from 11 rows to 121, which explains both the gain in coherence and the data sparsity (87.6% of contexts unseen).
5. *A test against the specification.* A Bayesian network requires that every CPT row be a probability distribution. That gave us the sum-to-1 test, the most valuable single test in this lab, since it caught the bug that sampling hid.

A sixth, smaller point: sampling each variable after its parents is exactly the left-to-right generation loop, so "generate text" and "sample from the network" are the same thing.

---

## 11. Reflection: how the LLM was used and how its output was checked (Deliverable 7)

**How I used it.** I wrote down the model first (variables, dependencies, estimator, what the program must compute), then gave the LLM a behavioural specification for the first-order model, then asked for the two generation modes, and finally asked for the second-order version after stating explicitly what must change in the probabilistic model. The prompts and the "what changes / what stays" table are in `prompts.md`. I did not accept code because it ran; I read it against the specification and tested it.

**Example of LLM-generated code that I inspected: the generation loop** (`generate()`, `src/first_order_lm.py`, lines 89-110)

```python
words = []
prev = START
while len(words) < max_len:
    if mode == "greedy":
        nxt = self.most_probable(prev)
    else:
        nxt = self.sample_next(prev, rng)
    if nxt is None or nxt == END:
        break
    words.append(nxt)
    prev = nxt
return words
```

What I checked, one point at a time:
1. *Does it stop?* The specification says "stop at `<END>`". Following Mode A by hand, `the -> cat -> sat -> on -> the -> cat ...` never reaches `<END>`, because `<END>` is not the arg max on that cycle. A loop that only stops on `<END>` would therefore run forever. The `while len(words) < max_len` guard is what prevents this, and the tests check it (`test_greedy_outputs`, `test_max_len_is_respected`). This case is not in the handout's prompt, so I added it explicitly to Prompt 2.
2. *Unseen contexts.* `nxt is None` handles a word with no row. Reading `most_probable` and `sample_next`, they use `self.distribution(prev)`, which uses `.get`, so there is no `KeyError` and no accidental new row (`test_lookup_does_not_add_rows`).
3. *Ties.* `max()` returns the first maximal item, and `distribution()` is sorted alphabetically, so a tie such as `cat` vs `dog` is broken the same way every time (`test_tie_is_broken_alphabetically`).

**Other places where the specification and the code had to agree**
- `sample_next` must use weighted `rng.choices`; `random.choice` would run and look plausible but ignore the CPT. `UniformSampler` in the tests confirms the frequency test catches it.
- Normalisation can be wrong without any visible symptom (Q8); only the sum-to-1 test finds it.
- In the second-order code, the two `<START>` pads are what let the first word have a complete context. `OneStartPad` in the tests shows that with a single pad `P(the | <START>, <START>)` becomes 0.

**What I take from this.** The code did what the prompt said. The subtle problems were all in things a prompt does not mention by default: loops, ties, unseen words and normalisation. Specifying the probabilistic model first, and then testing properties of that model, is what caught them.
