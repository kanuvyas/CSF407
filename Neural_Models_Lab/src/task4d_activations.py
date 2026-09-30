"""Task 4 Part D: same network/seed/optimiser, only the hidden activation changes."""
import os, json, statistics as st
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import X, Y_BIN, make_mlp, train

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
loss_fn = nn.BCEWithLogitsLoss()
STEPS, LR, EARLY = 5000, 1.0, 10        # gradient norm reported at step 0 and at step EARLY
ACTS = ["sigmoid", "tanh", "relu"]


def run(act, seed):
    m = make_mlp(act, seed=seed)
    h = train(m, X, Y_BIN, loss_fn, steps=STEPS, lr=LR)
    with torch.no_grad():
        ok = bool(((torch.sigmoid(m(X)) > 0.5).float() == Y_BIN).all())
    return m, h, ok


def diagnose(model, act):
    """Pre-activations a = W1 x + b1 and activations h for the 4 inputs."""
    with torch.no_grad():
        a = model[0](X); h = model[1](a)
    if act == "relu":
        dead = [bool((a[:, j] <= 0).all()) for j in range(a.shape[1])]
        return f"pre-act per unit (min,max): {[(round(a[:,j].min().item(),2), round(a[:,j].max().item(),2)) for j in range(2)]}; dead-on-all-inputs: {dead}"
    deriv = h * (1 - h) if act == "sigmoid" else 1 - h ** 2
    return f"mean local derivative per unit: {[round(deriv[:,j].mean().item(),4) for j in range(2)]} (max possible {0.25 if act=='sigmoid' else 1.0})"


results = {}
for seed in [0, 27]:
    print(f"\n==================== seed {seed} (identical initial weights for all three) ====================")
    print(f"{'act':8s} | {'final loss':>10s} | 4/4 correct | ||dL/dW1|| step0 | ||dL/dW1|| step{EARLY}")
    for act in ACTS:
        m, h, ok = run(act, seed)
        print(f"{act:8s} | {h['final_loss']:10.5f} | {str(ok):11s} | {h['gnorm_w1'][0]:.5f}        | {h['gnorm_w1'][EARLY]:.5f}")
        results[f"seed{seed}_{act}"] = dict(final_loss=h["final_loss"], all_correct=ok,
                                            gnorm_step0=h["gnorm_w1"][0], gnorm_early=h["gnorm_w1"][EARLY])
    print("-- diagnosis at INITIALISATION (this is where the early gradient is measured) --")
    for act in ACTS:
        print(f"  {act:8s}: {diagnose(make_mlp(act, seed=seed), act)}")
    print("-- diagnosis after training --")
    for act in ACTS:
        m, h, ok = run(act, seed)
        with torch.no_grad():
            pr = [round(v, 3) for v in torch.sigmoid(m(X)).squeeze().tolist()]
        print(f"  {act:8s}: {diagnose(m, act)}\n            probabilities {pr}")

# ---- aggregate over many seeds (repeated-run behaviour) --------------------
N = 40
print(f"\n==================== {N} seeds, same protocol ====================")
print(f"{'act':8s} | success | median final loss | median ||dL/dW1|| step0 | median ||dL/dW1|| step{EARLY}")
agg = {}
for act in ACTS:
    oks, fl, g0, ge = [], [], [], []
    for s in range(N):
        m, h, ok = run(act, s)
        oks.append(ok); fl.append(h["final_loss"]); g0.append(h["gnorm_w1"][0]); ge.append(h["gnorm_w1"][EARLY])
    agg[act] = dict(success=sum(oks), n=N, median_final_loss=st.median(fl),
                    median_g0=st.median(g0), median_gE=st.median(ge))
    print(f"{act:8s} | {sum(oks):2d}/{N}   | {st.median(fl):.5f}           | {st.median(g0):.5f}                 | {st.median(ge):.5f}")
results["aggregate"] = agg
json.dump(results, open(os.path.join(OUT, "activations.json"), "w"), indent=2)

# ---- plot: loss curves for seed 0 ----------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for seed, a in zip([0, 27], ax):
    for act in ACTS:
        _, h, _ = run(act, seed)
        a.plot(h["loss"], label=act)
    a.set_yscale("log"); a.set_title(f"seed {seed}"); a.set_xlabel("step"); a.grid(alpha=0.3); a.legend()
ax[0].set_ylabel("BCE loss")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "activation_loss_curves.png"), dpi=150)
