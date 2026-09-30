"""Task 5: same input, three classes. Only the output layer (3 logits) and the loss change."""
import os, json
import torch, torch.nn as nn
from common import X, Y_CLS, make_mlp, train

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED, ACT, HIDDEN, STEPS, LR = 0, "tanh", 2, 5000, 0.5
loss_fn = nn.CrossEntropyLoss()         # log-softmax + negative log-likelihood, takes raw logits

model = make_mlp(ACT, n_hidden=HIDDEN, n_out=3, seed=SEED)
hist = train(model, X, Y_CLS, loss_fn, steps=STEPS, lr=LR)

print(f"hidden units = {HIDDEN}, hidden act = {ACT}, SGD lr = {LR}, {STEPS} steps, seed {SEED}")
print("shape of final weight matrix W2 :", tuple(model[2].weight.shape), "(3 classes x 2 hidden units)")
with torch.no_grad():
    logits = model(X)
    probs = torch.softmax(logits, dim=1)
print("logits per example              :", logits.shape[1])
print("initial loss / final loss       :", round(hist["loss"][0], 4), "/", round(hist["final_loss"], 6))
print("\ninput   -> class probabilities [P(0), P(1), P(2)]   predicted  target")
for x, p, y in zip(X.int().tolist(), probs, Y_CLS.tolist()):
    print(f"{x} -> {[round(v, 4) for v in p.tolist()]}   {int(p.argmax())}          {y}")
acc = bool((probs.argmax(1) == Y_CLS).all())
print("all 4 correct:", acc)

# ---- softmax checks for one example ----------------------------------------
i = 1
z, p = logits[i], probs[i]
print(f"\nExample {X[i].int().tolist()}: logits = {z.tolist()}")
print("softmax vector      :", p.tolist())
print("sum of components   :", p.sum().item())

# shift invariance: add 100 to every logit
p_shift = torch.softmax(z + 100.0, dim=0)
print("softmax(z + 100)    :", p_shift.tolist())
print("max |difference|    :", (p - p_shift).abs().max().item())

# naive softmax overflows for large logits; max-subtracted version does not
def softmax_naive(v):  return torch.exp(v) / torch.exp(v).sum()
def softmax_stable(v): e = torch.exp(v - v.max()); return e / e.sum()
print("\nfloat32 exp() overflows above ~88.7, so the naive formula breaks for large logits:")
print("naive  softmax(z+100) :", softmax_naive(z + 100.0).tolist())
print("stable softmax(z+100) :", softmax_stable(z + 100.0).tolist())
print("stable softmax(z+1000):", softmax_stable(z + 1000.0).tolist())

# ---- logit gradient equals p - y (per example) -----------------------------
z = logits.clone().detach().requires_grad_(True)
nn.CrossEntropyLoss(reduction="sum")(z, Y_CLS).backward()          # sum => per-example gradients
onehot = torch.nn.functional.one_hot(Y_CLS, 3).float()
print("\nautograd dL/dlogits (sum-reduced):\n", z.grad)
print("p - y:\n", probs - onehot)
print("max |autograd - (p - y)| =", (z.grad - (probs - onehot)).abs().max().item())

json.dump({"hidden": HIDDEN, "final_loss": hist["final_loss"], "all_correct": acc,
           "probs": probs.tolist()}, open(os.path.join(OUT, "three_class.json"), "w"), indent=2)
