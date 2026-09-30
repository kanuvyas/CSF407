"""Task 4 Part C: all-zero initialisation keeps the two hidden units identical forever."""
import os
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import X, Y_BIN, make_mlp, train

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
loss_fn = nn.BCEWithLogitsLoss()


def zero_model(act):
    m = make_mlp(act, seed=0)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    return m


for act in ["sigmoid", "tanh"]:
    m = zero_model(act)
    hist = train(m, X, Y_BIN, loss_fn, steps=5000, lr=1.0, record=True)
    print(f"\n===== zero init, hidden activation = {act} =====")
    print("step | W1 row 0          | W1 row 1          | max|row0-row1| | dL/dW1 norm | loss")
    for t in [0, 1, 2, 3, 10, 100, 1000, 4999]:
        w = hist["w1"][t]
        print(f"{t:4d} | {w[0].tolist()!s:17.17} | {w[1].tolist()!s:17.17} | "
              f"{(w[0]-w[1]).abs().max().item():.2e}       | {hist['gnorm_w1'][t]:.2e}    | {hist['loss'][t]:.4f}")
    with torch.no_grad():
        p = torch.sigmoid(m(X)).squeeze()
    print("final loss :", round(hist["final_loss"], 4))
    print("final probs:", [round(v, 3) for v in p.tolist()], "-> labels", (p > 0.5).int().tolist())
    print("rows identical at the end:", bool(torch.equal(m[0].weight[0], m[0].weight[1])))
    print("final W1:\n", m[0].weight.data)

# contrast: same architecture, random init, same number of steps
m = make_mlp("sigmoid", seed=0)
hist = train(m, X, Y_BIN, loss_fn, steps=5000, lr=1.0)
print("\n===== contrast: random init, sigmoid =====")
print("final loss:", round(hist["final_loss"], 4), "| rows differ by", (m[0].weight[0]-m[0].weight[1]).abs().max().item())

# ---- extra variant: identical NON-zero rows (weights actually move) -------
def identical_model(act):
    m = make_mlp(act, seed=3)
    with torch.no_grad():
        m[0].weight[1] = m[0].weight[0]; m[0].bias[1] = m[0].bias[0]   # hidden unit 2 = copy of unit 1
        m[2].weight[0, 1] = m[2].weight[0, 0]                          # equal outgoing weights
    return m

m = identical_model("tanh")
hist_id = train(m, X, Y_BIN, loss_fn, steps=5000, lr=1.0, record=True)
print("\n===== extra: identical (non-zero) hidden units, tanh =====")
print("step | W1 row 0               | W1 row 1               | max|row0-row1| | dL/dW1 norm | loss")
for t in [0, 1, 10, 100, 1000, 4999]:
    w = hist_id["w1"][t]
    print(f"{t:4d} | {[round(v,4) for v in w[0].tolist()]!s:22} | {[round(v,4) for v in w[1].tolist()]!s:22} | "
          f"{(w[0]-w[1]).abs().max().item():.2e}       | {hist_id['gnorm_w1'][t]:.2e}    | {hist_id['loss'][t]:.4f}")
with torch.no_grad():
    p = torch.sigmoid(m(X)).squeeze()
print("final loss :", round(hist_id["final_loss"], 4))
print("final probs:", [round(v, 3) for v in p.tolist()], "-> labels", (p > 0.5).int().tolist())
print("rows identical at the end:", bool(torch.equal(m[0].weight[0], m[0].weight[1])))

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].plot(hist_id["loss"], label="identical non-zero rows (moves, still stuck)")
ax[0].plot(hist["loss"], label="random init (contrast, sigmoid)")
ax[0].axhline(0.6931, ls="--", c="gray")
ax[0].set_xlabel("step"); ax[0].set_ylabel("loss"); ax[0].legend(fontsize=7); ax[0].set_title("Loss")
d_id = [(w[0] - w[1]).abs().max().item() for w in hist_id["w1"]]
ax[1].plot(d_id, label="identical init"); ax[1].set_ylim(-1e-3, 1e-3)
ax[1].set_title("max |row0 - row1| of W1 (exactly 0 forever)"); ax[1].set_xlabel("step")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "symmetry.png"), dpi=150)
