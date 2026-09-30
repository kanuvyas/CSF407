# LLM prompts and corrections

LLM assistant used: Claude. The design (Task 2 in `REPORT.md`) was written first; the prompts below
only ask the assistant to turn that design into code.

## Prompt 1: binary XOR implementation (Task 3)

> Generate minimal PyTorch code for the following model and dataset. Do not change the architecture
> or the task.
>
> Dataset (full batch, 4 examples), inputs (x1, x2) -> label y:
> (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0.
>
> Model: 2 inputs -> 2 hidden units -> 1 output, hidden activation tanh, random weight
> initialisation. Output should be a single logit trained with `BCEWithLogitsLoss` (mean over the 4
> examples). Plain full-batch SGD, a few thousand CPU steps, fixed random seed.
>
> After training, print: initial and final loss, all four probabilities, the thresholded labels, and
> the gradient tensor of the first-layer weight matrix after `backward()`. Also add a
> finite-difference check of that gradient in float64. Explain each test in one sentence, and mark
> in comments where the forward pass, the scalar loss, `backward()` and the optimiser step happen.

## Prompt 2: symmetry and activation experiments (Task 4 C/D)

> Using the same model, (C) make a copy where every weight and bias is zero, train it, and print the
> two rows of the first-layer weight matrix at several steps. (D) Train the randomly initialised
> network three times changing ONLY the hidden activation (sigmoid, tanh, ReLU), same seed, same
> optimiser. For each run report the final loss, whether all 4 examples are correct, and the
> Euclidean norm of dL/dW1 at an early step. Do not change anything else.

## Prompt 3: three-class extension (Task 5)

> Modify ONLY the output layer and the loss of the previous code: three logits, `CrossEntropyLoss`
> with integer labels 0 = (0,0), 1 = (0,1)/(1,0), 2 = (1,1). Keep the hidden layer as it was. Print
> the class probabilities for all four inputs, and for one example print the softmax vector and its
> sum. Verify numerically that the logit gradient equals p - y, and that adding 100 to all logits
> leaves the softmax unchanged.

## Corrections and changes I made to the generated code

1. **Initialisation.** The first draft used PyTorch's default `nn.Linear` initialisation. For a layer
   with only 2 units the weights are tiny and the hidden units barely differ, so I switched both
   layers to N(0, 1) weights. This is an engineering setting, not a change to the task.
2. **Learning rate and steps.** I compared SGD with lr 0.5, 1.0, 2.0 and Adam over 20 seeds. The
   success rate barely changed (see the report), so I kept plain SGD with lr = 1.0 and 5000 steps
   for the binary task. The failures are caused by bad initialisations, not by too few steps.
3. **Zero-initialisation (Part C).** The draft only showed the zero-init run, where nothing moves at
   all (loss stays at ln 2 and the gradient is exactly 0). That is correct but hides the mechanism,
   so I added a second run with identical *non-zero* hidden units, where the weights do change but
   the two rows stay identical.
4. **Softmax demo (Task 5).** My first print statement for the naive softmax had a confusing comment
   about `exp(100)`. In float32 `exp` overflows above about 88.7, so the naive version already gives
   NaN at +100. I fixed the comment and compared it with the max-subtracted version.
5. **Three-class hidden size.** I first used 3 hidden units, then checked 20 seeds and found 2 hidden
   units work every time for this task, so I kept 2 to match the binary architecture.
