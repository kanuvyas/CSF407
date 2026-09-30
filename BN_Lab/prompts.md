# Prompts given to the LLM (Parts V, X and XII)

The handout says the LLM must be given a **behavioural specification** of the
probabilistic model, not "write a language model". Before each prompt I wrote down
what the model is, so I could check the code against it afterwards.

---

## Step 0. What I decided before asking (Understand, Design)

| Question from Section 3 of the handout | My answer |
|---|---|
| What do the variables represent? | `X_t` is the t-th token of a sentence. Tokens are lower-case words plus `<START>` and `<END>`. |
| What are the dependencies? | Each token depends only on the token before it: `X1 -> X2 -> ... -> XT`. |
| What distribution is estimated? | `P(X_t \| X_{t-1}) = C(X_{t-1}, X_t) / sum_k C(X_{t-1}, w_k)` |
| What should the program compute? | counts, the CPT, next-word prediction, and sentence generation by repeated sampling until `<END>` |

---

## Prompt 1: first-order model (Part V)

> Write a simple Python implementation of a first-order autoregressive language model.
> The model should:
> 1. take a list of tokenised sentences as training data;
> 2. count transitions between consecutive tokens;
> 3. construct the conditional distribution P(X_t | X_{t-1});
> 4. display the probabilities for a specified previous token;
> 5. predict the most probable next token;
> 6. generate a sentence by repeatedly sampling the next token;
> 7. stop when the `<END>` token is generated.
>
> Do not use a machine-learning library or a pretrained language model. Use ordinary
> Python data structures and random sampling.

**Where each requirement ended up in `src/first_order_lm.py`**

| Requirement | Code |
|---|---|
| 1. tokenised sentences in | `fit(sentences)` |
| 2. count transitions | `self.counts[prev][nxt] += 1` (line 37) |
| 3. build P(X_t \| X_{t-1}) | `n / total` (line 43) |
| 4. show probabilities for one token | `distribution(prev)` |
| 5. most probable next token | `most_probable(prev)` |
| 6. generate by repeated sampling | `sample_next()` and `generate(mode="sample")` |
| 7. stop at `<END>` | `if nxt is None or nxt == END: break` in `generate()` |

## Prompt 2: two generation modes (Part X)

> Extend `generate()` so it supports Mode A (always take arg max P(w | previous word))
> and Mode B (sample from P(w | previous word)). Make the arg max deterministic when
> two words have the same probability. Make sure generation cannot run forever.

The last two sentences were added on purpose: they describe cases the handout does
not mention (ties, and greedy decoding that never reaches `<END>`).

---

## Prompt 3: second-order model (Part XII)

Before accepting any code I wrote down what has to change in the probabilistic model
and what has to stay the same.

**What changes**

| | First order | Second order |
|---|---|---|
| Graph | `X_{t-1} -> X_t` | `X_{t-2} -> X_t <- X_{t-1}` (two parents) |
| Context | one word | a pair `(X_{t-2}, X_{t-1})` |
| Counts | `C(a, b)` | `C(a, b, c)`, counts of observed triples |
| CPT | `C(a, b) / sum_k C(a, w_k)` | `C(a, b, c) / sum_k C(a, b, w_k)` |
| Padding | one `<START>` | two `<START>` tokens, so the first word has a full context |
| Generation | context = last word | context = last two words, window slides one step each time |

**What must not change:** `<END>` still stops generation, sampling is still a weighted
draw, still no ML library and no neural network.

**What I expected to see before running it**

1. up to 11 x 11 = 121 possible contexts, but only a few observed;
2. because the six sentences are so few, the model will mostly copy them;
3. `P("the cat sat on the park")` becomes 0, because `(on, the)` was only followed by `mat` and `rug`.

> Modify the existing first-order autoregressive model into a second-order model.
> The model should estimate P(X_t | X_{t-2}, X_{t-1}). Represent the model using counts
> of observed triples and use these counts to construct conditional probability
> distributions. Pad each sentence with two `<START>` tokens. Keep the same interface
> (`fit`, `distribution`, `most_probable`, `sample_next`, `generate`). Do not replace the
> model with a neural network or a pretrained language model.

All three expectations were confirmed by `results/model_comparison.md`
(15 of 121 contexts observed, 0 new sentences in 2000 samples, `P(...park) = 0`).
