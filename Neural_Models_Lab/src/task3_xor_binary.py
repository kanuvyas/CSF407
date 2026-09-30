"""Tasks 3 + 4 (Parts A and B): 2-2-1 network, tanh hidden, logits + BCEWithLogitsLoss."""
import os, json
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import X, Y_BIN, make_mlp, train

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
SEED, ACT, STEPS, LR = 0, "tanh", 5000, 1.0
loss_fn = nn.BCEWithLogitsLoss()           # mean over the 4 examples

model = make_mlp(ACT, n_hidden=2, n_out=1, seed=SEED)
hist = train(model, X, Y_BIN, loss_fn, steps=STEPS, lr=LR)

with torch.no_grad():
    probs = torch.sigmoid(model(X)).squeeze()
    labels = (probs > 0.5).int()
correct = bool((labels.float() == Y_BIN.squeeze()).all())

print(f"=== Part A: learning check (act={ACT}, seed={SEED}, SGD lr={LR}, {STEPS} full-batch steps) ===")
print(f"initial loss : {hist['loss'][0]:.4f}")
print(f"final loss   : {hist['final_loss']:.6f}")
print("inputs       :", X.int().tolist())
print("probabilities:", [round(p, 4) for p in probs.tolist()])
print("labels       :", labels.tolist(), " target:", Y_BIN.squeeze().int().tolist())
print("all 4 correct:", correct)

# ---- Part B: gradient inspection ---------------------------------------
print("\n=== Part B: gradient of the first layer ===")
fresh = make_mlp(ACT, n_hidden=2, n_out=1, seed=SEED)       # same initial weights
fresh.zero_grad(); loss_fn(fresh(X), Y_BIN).backward()
g_mean = fresh[0].weight.grad.clone()
print("W1 (initial)            :\n", fresh[0].weight.data)
print("W1.grad = dL/dW1 (mean loss):\n", g_mean)

# per-example gradients: the mean-loss gradient must equal their average
per_ex = []
for i in range(4):
    fresh.zero_grad(); loss_fn(fresh(X[i:i+1]), Y_BIN[i:i+1]).backward()
    per_ex.append(fresh[0].weight.grad.clone())
avg = torch.stack(per_ex).mean(0)
print("average of 4 per-example gradients:\n", avg)
print("max |mean-loss grad - average of per-example grads| =", (g_mean - avg).abs().max().item())

# finite-difference check in float64
m64 = make_mlp(ACT, n_hidden=2, n_out=1, seed=SEED, dtype=torch.float64)
x64, y64 = X.double(), Y_BIN.double()
m64.zero_grad(); loss_fn(m64(x64), y64).backward()
analytic = m64[0].weight.grad.clone()
numeric = torch.zeros_like(analytic); eps = 1e-6
with torch.no_grad():
    for i in range(2):
        for j in range(2):
            w = m64[0].weight
            old = w[i, j].item()
            w[i, j] = old + eps; lp = loss_fn(m64(x64), y64).item()
            w[i, j] = old - eps; lm = loss_fn(m64(x64), y64).item()
            w[i, j] = old
            numeric[i, j] = (lp - lm) / (2 * eps)
print("finite-difference dL/dW1:\n", numeric)
print("max |autograd - finite diff| =", (analytic - numeric).abs().max().item())

print("\nTrained W1:\n", model[0].weight.data, "\nTrained b1:", model[0].bias.data)
print("Trained W2:", model[2].weight.data, " b2:", model[2].bias.data)
with torch.no_grad():
    print("Hidden activations h = tanh(W1 x + b1) for the 4 inputs:\n", torch.tanh(model[0](X)))

json.dump({"seed": SEED, "act": ACT, "steps": STEPS, "lr": LR,
           "initial_loss": hist["loss"][0], "final_loss": hist["final_loss"],
           "probs": probs.tolist(), "labels": labels.tolist(), "all_correct": correct,
           "grad_fd_max_err": (analytic - numeric).abs().max().item()},
          open(os.path.join(OUT, "xor_binary.json"), "w"), indent=2)

# ---- plots -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].plot(hist["loss"]); axes[0].set_yscale("log")
axes[0].set_xlabel("step"); axes[0].set_ylabel("BCE loss"); axes[0].set_title("Training loss (tanh 2-2-1)")
axes[0].grid(alpha=0.3)
g = torch.linspace(-0.3, 1.3, 200)
xx, yy = torch.meshgrid(g, g, indexing="ij")
with torch.no_grad():
    zz = torch.sigmoid(model(torch.stack([xx.flatten(), yy.flatten()], 1))).reshape(200, 200)
cs = axes[1].contourf(xx, yy, zz, levels=20, cmap="RdBu_r", alpha=0.8)
fig.colorbar(cs, ax=axes[1], label="P(warning)")
for (a, b), y in zip(X.tolist(), Y_BIN.squeeze().tolist()):
    axes[1].scatter(a, b, s=120, c="k", marker="o" if y else "s", edgecolor="w", zorder=3)
axes[1].set_title("Learned decision surface"); axes[1].set_xlabel("x1"); axes[1].set_ylabel("x2")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "xor_binary_training.png"), dpi=150)
