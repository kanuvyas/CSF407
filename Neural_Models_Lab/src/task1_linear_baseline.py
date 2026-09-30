"""Task 1: a single affine map + sigmoid cannot solve XOR. Also draws the 4-point sketch."""
import os
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import X, Y_BIN

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
torch.manual_seed(0)
lin = nn.Linear(2, 1)                      # affine map; sigmoid is inside the loss
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.SGD(lin.parameters(), lr=1.0)
for step in range(5000):
    opt.zero_grad(); loss = loss_fn(lin(X), Y_BIN); loss.backward(); opt.step()

with torch.no_grad():
    p = torch.sigmoid(lin(X)).squeeze()
print("Linear model (affine + sigmoid) on XOR")
print("final loss      :", round(loss.item(), 4), "(ln 2 = 0.6931 is the 'always say 0.5' loss)")
print("probabilities   :", [round(v, 3) for v in p.tolist()])
print("thresholded     :", (p > 0.5).int().tolist(), " target:", Y_BIN.squeeze().int().tolist())
print("n correct       :", int(((p > 0.5).float() == Y_BIN.squeeze()).sum()), "/ 4")
print("weights, bias   :", lin.weight.data.squeeze().tolist(), lin.bias.item())

# ---- sketch of the four points + one attempted straight line ------------
fig, ax = plt.subplots(figsize=(4.2, 4.2))
for (x1, x2), y in zip(X.tolist(), Y_BIN.squeeze().tolist()):
    ax.scatter(x1, x2, s=260, c="tab:red" if y else "tab:blue",
               marker="o" if y else "s", edgecolor="k", zorder=3)
    ax.annotate(f"({int(x1)},{int(x2)}) y={int(y)}", (x1, x2), textcoords="offset points",
                xytext=(0, 14), ha="center", fontsize=8)
xs = torch.linspace(-0.3, 1.3, 10)
ax.plot(xs, 0.5 + 0 * xs, "--", color="gray", lw=1)
ax.plot(xs, 1.0 - xs + 0.0, "--", color="gray", lw=1)
ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.4)
ax.set_xlabel("x1"); ax.set_ylabel("x2")
ax.set_title("XOR: red circles = 1, blue squares = 0\n(two example straight lines, both fail)", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "task1_xor_points.png"), dpi=150)
