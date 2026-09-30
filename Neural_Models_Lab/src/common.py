"""Shared data, model builders and a small training loop for the XOR lab."""
import torch
import torch.nn as nn

# ---- the four sensor examples (x1, x2) -> label -------------------------
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y_BIN = torch.tensor([[0.], [1.], [1.], [0.]])      # XOR: disagreement warning
Y_CLS = torch.tensor([0, 1, 1, 2])                  # 3-class: none / disagree / both

ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}


def make_mlp(act="tanh", n_hidden=2, n_out=1, seed=0, dtype=torch.float32):
    """2 -> n_hidden -> n_out network. Output is raw logits (no sigmoid/softmax
    layer) because BCEWithLogitsLoss / CrossEntropyLoss apply it internally in a
    numerically stable way."""
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(2, n_hidden),
        ACTIVATIONS[act](),
        nn.Linear(n_hidden, n_out),
    )
    # default PyTorch init is very small for a 2-unit layer; N(0,1) weights give
    # the hidden units a real chance to become different from each other.
    with torch.no_grad():
        for layer in (model[0], model[2]):
            nn.init.normal_(layer.weight, std=1.0)
            nn.init.normal_(layer.bias, std=0.1)
    return model.to(dtype)


def first_layer_grad_norm(model):
    return model[0].weight.grad.norm().item()


def train(model, x, y, loss_fn, steps=3000, lr=0.5, record=False):
    """Full-batch gradient descent. Returns a history dict.

    history["loss"][t]      loss *before* the update at step t (t = 0 is the initial loss)
    history["gnorm_w1"][t]  Euclidean norm of dL/dW1 at step t
    history["w1"][t]        copy of the first-layer weight matrix (only if record=True)
    """
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    hist = {"loss": [], "gnorm_w1": [], "w1": []}
    for _ in range(steps):
        opt.zero_grad()
        logits = model(x)                # forward pass
        loss = loss_fn(logits, y)        # scalar loss
        loss.backward()                  # reverse-mode autodiff (backprop)
        hist["loss"].append(loss.item())
        hist["gnorm_w1"].append(first_layer_grad_norm(model))
        if record:
            hist["w1"].append(model[0].weight.detach().clone())
        opt.step()                       # parameter update
    with torch.no_grad():                # loss after the final update
        hist["final_loss"] = loss_fn(model(x), y).item()
    return hist
