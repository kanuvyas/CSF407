# Laboratory Report: Neural Models (Learning, Depth, Activations, Output Layers)

All numbers below come from `run_all.sh` (logs in `results/`). Everything uses fixed seeds and runs on CPU.

---

## Task 1: Problem specification

**Input space:** X = {0, 1}², the two binary sensor readings (x1, x2).
**Output space:** Y = {0, 1}, where 1 means "raise the sensor-disagreement warning".

| x1 | x2 | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

**Sketch** (`results/task1_xor_points.png`): the two class-1 points (0,1) and (1,0) sit on one diagonal of the unit square and the two class-0 points (0,0) and (1,1) on the other.

**Why one line is not enough.** A straight line splits the plane into two half-planes. The midpoint of (0,1) and (1,0) and the midpoint of (0,0) and (1,1) are the same point, (0.5, 0.5). If a line put both class-1 points on one side, then their midpoint would also be on that side (half-planes are convex). The same argument applies to the two class-0 points, so (0.5, 0.5) would have to be on both sides at once, which is impossible.

**Prediction for a single affine layer + sigmoid.** I expected it to learn nothing useful: the best it can do is output about 0.5 for every input, so the loss should sit at ln 2 ≈ 0.693 and it can get at most 2 of 4 right.

**Result** (`results/task1_linear_baseline.txt`): final loss 0.6931, all four probabilities 0.5, and both weights ≈ 1e-7 (i.e. zero). So the prediction was right. The data is perfectly balanced and symmetric, so the gradient for the weights cancels out and the model just learns the constant "0.5".

**Think about it: what claim about representation does XOR test?** That a representation that is linear in the raw inputs is not enough for some problems, and that a hidden layer with a nonlinearity can build new features in which the classes *do* become linearly separable. With four points we can check this exactly: the linear model fails completely, and a network with two hidden units succeeds.

---

## Task 2: Model design and validation criteria

**Design:** 2 inputs → 2 hidden units (tanh) → 1 output logit; sigmoid is applied to the logit to get a probability, loss = binary cross-entropy (in PyTorch: `BCEWithLogitsLoss`), full-batch gradient descent (SGD, lr = 1.0, 5000 steps). Later I repeat the experiment with sigmoid and ReLU hidden units.

1. **Why the hidden nonlinearity is necessary.** Without it, the hidden layer and output layer are two affine maps, and two affine maps compose into one affine map (`W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2)`; checked in `tests/test_lab.py`). So the model would be no more expressive than the linear model from Task 1, no matter how many layers I stack. The nonlinearity lets the hidden units carve the plane with bent/parallel boundaries and produce new features.
2. **Why sigmoid + binary cross-entropy.** The target is one yes/no answer, so the output should be a probability in (0, 1), which is what sigmoid gives. Cross-entropy is the negative log-likelihood of a Bernoulli variable, and with this pairing the gradient with respect to the logit is just (p − y): it does not vanish when the model is confidently wrong, unlike squared error on a sigmoid output.
3. **Validation criteria** (decided before running):
   * (a) final loss is far below ln 2 (I required < 0.01) and much lower than the initial loss;
   * (b) all four thresholded predictions match the labels, with probabilities near 0 / 1;
   * (c) the first-layer gradient is nonzero and matches a finite-difference estimate (so backprop is computing the right thing);
   * (d) the result is not a fluke of one seed: I repeat with many seeds and report the success rate;
   * (e) the learned hidden units should be *different* from each other and interpretable.

**Think about it: who decides what the hidden units compute?** Nobody does directly. Only the output has a target. The loss gradient flows back through the output weights to each hidden unit (∂L/∂h = W2ᵀ · ∂L/∂logit), so each hidden unit is nudged toward whatever feature makes the final answer better. In my run they ended up as an OR-like unit and an AND-like unit (see Task 4A), and the output layer computes "OR and not AND", which is XOR.

---

## Task 3: LLM-generated implementation

Prompts are in `prompts.md`; code is in `src/` (`common.py` + `task3_xor_binary.py`).

**Where things happen in the code (`common.train`):**
* forward pass: `logits = model(x)`
* scalar loss: `loss = loss_fn(logits, y)` (mean over the four examples)
* reverse-mode autodiff: `loss.backward()`
* parameter update: `opt.step()` (plain SGD)

**Changes I made before accepting the code:** (1) switched from PyTorch's default tiny initialisation to N(0,1) weights, since two hidden units with tiny weights start almost identical; (2) added a float64 finite-difference check and a per-example-gradient check of the first-layer gradient. Full list in `prompts.md`.

**Think about it: what can be verified without running?** From reading the code I can confirm the architecture (layer sizes), that the labels are the XOR labels, that the loss is applied to logits (not sigmoid twice), that `zero_grad()` comes before `backward()`, and that the optimiser is given all parameters. What needs execution and measurement: whether the loss actually drops, whether the gradients are nonzero and numerically correct, whether it is seed-sensitive, and numerical problems like overflow or NaNs.

---

## Task 4: Execution and diagnosis

### Part A: basic learning check (tanh, seed 0, SGD lr 1.0, 5000 steps)

| | value |
|---|---|
| initial loss | 0.8309 |
| final loss | 0.000618 |
| probabilities for (0,0), (0,1), (1,0), (1,1) | 0.0008, 0.9996, 0.9996, 0.0008 |
| thresholded labels | 0, 1, 1, 0 (all 4 correct) |

Trained first layer: row 1 = [3.42, 3.42], bias −5.03; row 2 = [4.21, 4.20], bias −2.03. So hidden unit 1 turns on only when *both* sensors are on (AND-like), hidden unit 2 turns on when *at least one* is on (OR-like). The output weights are about −8.06 and +7.99, so the output is roughly "OR and not AND". That is a readable solution and not just a lucky number. Figure: `results/xor_binary_training.png` shows the loss curve and the learned decision surface, which is a diagonal band (two parallel boundaries), something no single line could do.

### Part B: backpropagation check

At the initial weights the first-layer gradient is

```
dL/dW1 = [[ 0.1392, 0.0222],
          [-0.2904, 0.1037]]
```

`W1.grad` holds ∂L/∂W(1): entry (i, j) says how much the loss changes per small increase of the weight from input j into hidden unit i. A gradient step moves W1 against this.

* The loss is the **mean** over the four examples, L = (1/4) Σ Lₙ. Differentiation is linear, so ∂L/∂W = (1/4) Σ ∂Lₙ/∂W, i.e. the average of the per-example gradients. I checked this numerically: the maximum difference between the mean-loss gradient and the average of four separate single-example gradients is 0.0.
* I also compared autograd to central finite differences in float64: maximum difference 8.1e-11, so backprop is computing the right derivative.

### Part C: symmetry experiment

**Exactly zero init** (all weights and biases = 0, both for sigmoid and tanh): the two rows of W1 stay [0, 0] and [0, 0] at every recorded step (0, 1, 2, 3, 10, 100, 1000, 4999), the loss stays at 0.6931 and the gradient norm of W1 is exactly 0. Final probabilities are all 0.5.

This is actually a slightly stronger failure than "the rows stay the same". Because W2 = 0, the error signal that reaches the hidden layer, W2ᵀ · δ, is zero, so W1 gets no gradient. And since XOR's labels are balanced, the gradient on W2 and b2 also cancels to zero at p = 0.5. So the all-zero point is a stationary point: nothing moves at all.

**Identical but non-zero hidden units** (I added this variant to see the symmetry effect while the weights really do change): I made hidden unit 2 a copy of unit 1 (same weights and bias) with equal outgoing weights. Here the weights do move (row 1 goes from [0.80, 0.60] to [5.05, 5.05]) but `max|row0 − row1|` is exactly 0.0 at every step. The network ends at loss 0.478 with probabilities (0.001, 0.667, 0.667, 0.667), i.e. predictions 0, 1, 1, 1: it has effectively become a one-hidden-unit network computing OR, so 3 of 4 correct. Figure: `results/symmetry.png`.

**Explanation.** Two hidden units with the same incoming weights compute the same output for every input. Their outgoing weights are equal too, so each one influences the loss in exactly the same way, and backprop hands both the *same* gradient. Equal weights plus equal updates stay equal forever. The second hidden unit is therefore wasted capacity, and the network can never be more expressive than a one-unit hidden layer. Random initialisation breaks the tie (in the contrast run the rows end up differing by 0.36 and the loss goes to 0.0025).

### Part D: activation experiment

Everything is identical (architecture, loss, optimiser, steps, seed and initial weights) except the hidden activation. "Early gradient" is ‖∂L/∂W(1)‖₂ at step 0 (and at step 10 in the table).

**Seed 0**

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇W(1)L‖₂ (step 0) | step 10 |
|---|---|---|---|---|
| Sigmoid | 0.00247 | Yes | 0.0766 | 0.0168 |
| Tanh | 0.00062 | Yes | 0.3390 | 0.0906 |
| ReLU | 0.34663 | **No** | 0.2705 | 0.1095 |

**Seed 27** (a seed where all three succeed)

| Hidden activation | Final loss | 4/4 correct? | Early ‖∇W(1)L‖₂ (step 0) | step 10 |
|---|---|---|---|---|
| Sigmoid | 0.00395 | Yes | 0.0084 | 0.0025 |
| Tanh | 0.00062 | Yes | 0.0422 | 0.0264 |
| ReLU | 0.00028 | Yes | 0.0694 | 0.0511 |

**40 seeds** (same protocol, seeds 0-39)

| Hidden activation | Runs solving XOR | Median final loss | Median early gradient norm (step 0) |
|---|---|---|---|
| Sigmoid | 30 / 40 | 0.0021 | 0.0212 |
| Tanh | 15 / 40 | 0.3468 | 0.0598 |
| ReLU | 8 / 40 | 0.4774 | 0.0999 |

**Interpretation.** At initialisation the sigmoid run had the smallest gradient in both seeds (0.077 vs 0.34 for tanh at seed 0, and the 40-seed median is also the smallest). This matches the math: the sigmoid's derivative is at most 0.25, whereas tanh's is at most 1 and an active ReLU's is exactly 1. At initialisation the measured mean local derivatives were about 0.23-0.24 for sigmoid and 0.8-0.9 for tanh at seed 0. So, as a *mechanism*, sigmoid gives smaller gradients. But a smaller gradient did **not** mean a worse result here: sigmoid solved XOR most often (30/40), and ReLU, which had the largest gradients, solved it least often (8/40). So for this experiment the gradient size at the start does not predict success. What mattered more was whether the random start led into a bad partial solution. I have not tested why sigmoid was the most reliable; one possibility is that the larger steps of tanh/ReLU commit to a partial solution faster, but that is only a guess. With only four data points and 2 hidden units I would not generalise any of this to "sigmoid is better than ReLU".

The seed-0 ReLU failure has a clear cause (I checked the pre-activations). Inputs (0,0) and (1,0) have both ReLU pre-activations negative, so both hidden outputs are exactly 0, the logit equals the output bias (≈ 0), and p = 0.5. Because the ReLU derivative is 0 there, these two examples send no gradient into W1 at all, so the network cannot fix them. Probabilities at the end: (0.5, 1.0, 0.5, 0.0).

**Think about it: saturated sigmoid vs negative ReLU.** Look at the pre-activation a and the activation h. A saturated sigmoid/tanh unit has a *large |a|* (for example |a| > 5), h sits near 0/1 (or ±1), and its derivative h(1−h) is tiny but *not exactly* zero. A negative ReLU has a ≤ 0, h is *exactly* 0 and its derivative is *exactly* 0. A unit that is dead for every input never recovers, while a saturated one still gets a (very small) gradient. In my runs, after training the sigmoid units had mean derivative about 0.02 (saturated by design, because they confidently output 0/1 values), while the seed-0 ReLU units were not dead overall, only inactive for two specific inputs.

---

## Task 5: Three-class extension

**Change:** only the output layer and loss. The network is 2 → 2 (tanh) → **3 logits**, trained with `CrossEntropyLoss` (softmax + negative log-likelihood) on labels 0 = (0,0), 1 = (0,1) or (1,0), 2 = (1,1). SGD lr 0.5, 5000 steps, seed 0.

**Predictions made before running:**
1. The final weight matrix has shape (3, 2): 3 output classes × 2 hidden units.
2. There are 3 logits per example (one per class).
3. Softmax probabilities sum to one because softmax divides each exp(zₖ) by the sum of all of them: pₖ = exp(zₖ)/Σⱼ exp(zⱼ).
4. The logit gradient is p − y (derivation below).

**Results** (`results/task5_three_class.txt`): W2 has shape (3, 2), 3 logits per example, loss 1.500 → 0.0005, and all four inputs are classified correctly:

| input | P(class 0) | P(class 1) | P(class 2) | predicted |
|---|---|---|---|---|
| (0,0) | 0.9995 | 0.0005 | 0.0000 | 0 |
| (0,1) | 0.0002 | 0.9996 | 0.0002 | 1 |
| (1,0) | 0.0002 | 0.9996 | 0.0002 | 1 |
| (1,1) | 0.0000 | 0.0006 | 0.9994 | 2 |

For the example (0,1) the logits are (−3.64, 4.77, −3.73) and the softmax vector is (0.000221, 0.999577, 0.000202), which sums to 1.0 (to float32 precision).

**Why the gradient is p − y.** With L = −Σₖ yₖ log pₖ and pₖ = exp(zₖ)/Σⱼ exp(zⱼ), we get ∂L/∂zᵢ = −Σₖ yₖ (δₖᵢ − pᵢ) = −yᵢ + pᵢ Σₖ yₖ = pᵢ − yᵢ, since Σₖ yₖ = 1 for a one-hot label. I verified this numerically against autograd: maximum difference 1.2e-10.

**Optional diagnostic.** Adding 100 to every logit changes the softmax vector by at most 4e-10 (round-off only), because the common factor exp(100) cancels in the ratio. This matters for implementation: the naive formula `exp(z)/sum(exp(z))` overflows in float32 (exp overflows above about 88.7), and indeed it returns NaN for z + 100. Stable implementations subtract the maximum logit first: the largest exponent becomes exp(0) = 1, nothing overflows, and the result is mathematically identical thanks to the shift invariance. The stable version gives the correct vector even for z + 1000.

**A side observation.** This three-class task was solved by the 2-hidden-unit network on all 20 seeds I tried, unlike XOR. That makes sense because the classes are ordered by x1 + x2 ∈ {0, 1, 2}, so the task is much easier and doesn't need the OR/AND trick.

**Think about it: scaling to next-token prediction.** What stays the same mathematically is the output recipe: the network produces one logit per class, softmax turns them into a distribution, cross-entropy is −log of the probability of the correct class, the logit gradient is still p − y, and softmax is still shift-invariant, so stable implementations subtract the max. What changes dramatically is everything around it: the logits come from a very deep architecture (a Transformer with attention) instead of one hidden layer; the input is a long sequence of token embeddings instead of two bits; there are tens of thousands of classes, so the output matrix is huge and computing the softmax is a major cost; the data are billions of tokens trained in mini-batches on GPUs with Adam-like optimisers instead of four points with full-batch SGD; and the training and evaluation need to be done much more carefully (sampling, perplexity, etc.).

---

## Reflection Questions

**1. What did the XOR experiment demonstrate about depth vs nonlinearity?**
Depth alone is not enough; nonlinearity is what makes depth useful. A stack of affine layers is still one affine map, so it fails on XOR exactly like the linear model (loss stuck at 0.693). Once I put a tanh between the layers, two hidden units were enough to solve XOR, by learning an OR-like and an AND-like feature.

**2. What evidence showed that backpropagation gave a useful learning signal rather than a merely nonzero gradient?**
A nonzero gradient by itself proves little (the stuck identical-units run also had nonzero gradients for a while). The useful evidence is that (a) the autograd gradient matched finite differences to about 1e-10, so it was the correct derivative, (b) the loss went down steadily from 0.83 to 0.0006, (c) all four predictions flipped to the right labels, and (d) the two hidden units became different and meaningful (OR-like and AND-like), even though neither had a target of its own.

**3. Why did identical/zero weight initialisation stop the two hidden units from learning distinct features?**
Identical hidden units compute identical outputs and have identical outgoing weights, so backprop gives them identical gradients and they stay identical after every update. With all-zeros it is even more extreme: W2 = 0 means no error signal reaches W1 at all, and the balanced labels make the output-layer gradient cancel too, so nothing changes. Random initialisation breaks the symmetry.

**4. How did changing the hidden activation affect the gradient? (Scientific explanation vs engineering observation.)**
*Scientific:* the gradient to the first layer contains the local derivative f′(a) of the hidden activation as a factor. Sigmoid's derivative is at most 0.25, tanh's at most 1, and an active ReLU's is exactly 1 (and exactly 0 if inactive), so I expect smaller early gradients for sigmoid and possible blocked gradients for ReLU. *Engineering observation:* at seed 0 the early gradient norms were 0.077 (sigmoid), 0.339 (tanh), 0.270 (ReLU), consistent with that, and the 40-seed median for sigmoid was also the smallest. But gradient size did not predict success: sigmoid solved XOR 30/40 times, tanh 15/40, ReLU 8/40, and the ReLU failure at seed 0 was caused by two inputs with both units inactive, which is a zero-gradient effect. So the mechanism is well explained, but the ranking of activations is just an observation of this tiny experiment.

**5. Why must the output layer and loss be selected together according to the task?**
The output activation defines what the numbers mean (a probability for a yes/no answer, a distribution over K classes, a real number for regression), and the loss has to match that meaning. Sigmoid + BCE and softmax + cross-entropy are the negative log-likelihoods of Bernoulli and categorical distributions, and they give the clean logit gradient p − y, so the learning signal stays strong when the model is confidently wrong. A mismatched pair (for example softmax with squared error, or a linear output with BCE) gives meaningless probabilities, weaker gradients, or numerical problems. In PyTorch it's also why the loss functions take raw logits: the stable log-softmax / log-sigmoid is fused into the loss.

**6. One example where the LLM improved productivity and one where human verification was essential.**
*Productivity:* the LLM wrote the PyTorch boilerplate (model, training loop, finite-difference check, plotting and the result tables) very quickly, which left time for the experiments and interpretation. *Verification was essential:* in the zero-initialisation experiment the plain "rows stay identical" story hides that nothing moves at all, so I checked the gradients and found the W2 = 0 / balanced-label effect and added a non-zero identical-units run to actually show the symmetry. Another one was the naive softmax comment about exp(100), which was wrong until I ran it (float32 already overflows). I also had to check the default initialisation, which was too small for a two-unit network.

**7. Which tests would I keep at scale, and which would become too expensive?**
Keep: monitoring the loss and predictions on held-out data, checking that gradients are finite and nonzero (and their norms per layer), a tiny-batch overfitting test (can the model memorise 4-32 examples?), shape and softmax-sum/shift-invariance checks, unit tests on small versions of the model, and repeating with several seeds if affordable. Too expensive: exhaustive finite-difference gradient checks (they need two forward passes per parameter, which is impossible with billions of parameters), so at scale I would only check gradients on a tiny copy of the model or a few randomly picked parameters; also running dozens of seeds for every configuration, and printing or inspecting all activations.

---

## Reproducing

```bash
pip install -r requirements.txt
./run_all.sh
```
