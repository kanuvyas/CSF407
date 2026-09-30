"""Automated checks for the claims made in REPORT.md. Run with:  pytest -q"""
import torch, torch.nn as nn
from common import X, Y_BIN, Y_CLS, make_mlp, train

bce = nn.BCEWithLogitsLoss()


def test_affine_stack_collapses_to_one_affine_map():
    torch.manual_seed(1)
    l1, l2 = nn.Linear(2, 5).double(), nn.Linear(5, 1).double()
    W = l2.weight @ l1.weight                      # combined weight
    b = l2.weight @ l1.bias + l2.bias              # combined bias
    x = torch.randn(10, 2).double()
    assert torch.allclose(l2(l1(x)), x @ W.T + b)


def test_single_affine_plus_sigmoid_cannot_solve_xor():
    torch.manual_seed(0)
    lin = nn.Linear(2, 1)
    opt = torch.optim.SGD(lin.parameters(), lr=1.0)
    for _ in range(3000):
        opt.zero_grad(); bce(lin(X), Y_BIN).backward(); opt.step()
    pred = (torch.sigmoid(lin(X)) > 0.5).float()
    assert int((pred == Y_BIN).sum()) < 4
    assert bce(lin(X), Y_BIN).item() > 0.69          # stuck at ln 2


def test_hidden_nonlinearity_learns_xor():
    m = make_mlp("tanh", seed=0)
    h = train(m, X, Y_BIN, bce, steps=5000, lr=1.0)
    pred = (torch.sigmoid(m(X)) > 0.5).float()
    assert torch.equal(pred, Y_BIN)
    assert h["final_loss"] < 0.01 < h["loss"][0]


def test_autograd_matches_finite_differences():
    m = make_mlp("tanh", seed=0, dtype=torch.float64)
    x, y = X.double(), Y_BIN.double()
    m.zero_grad(); bce(m(x), y).backward()
    w = m[0].weight
    analytic = w.grad.clone()
    eps = 1e-6
    with torch.no_grad():
        for i in range(2):
            for j in range(2):
                old = w[i, j].item()
                w[i, j] = old + eps; lp = bce(m(x), y).item()
                w[i, j] = old - eps; lm = bce(m(x), y).item()
                w[i, j] = old
                assert abs((lp - lm) / (2 * eps) - analytic[i, j].item()) < 1e-7


def test_mean_loss_gradient_is_average_of_per_example_gradients():
    m = make_mlp("sigmoid", seed=2)
    m.zero_grad(); bce(m(X), Y_BIN).backward()
    g = m[0].weight.grad.clone()
    per = []
    for i in range(4):
        m.zero_grad(); bce(m(X[i:i+1]), Y_BIN[i:i+1]).backward()
        per.append(m[0].weight.grad.clone())
    assert torch.allclose(g, torch.stack(per).mean(0), atol=1e-7)


def test_zero_init_keeps_hidden_units_identical():
    m = make_mlp("sigmoid", seed=0)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
    train(m, X, Y_BIN, bce, steps=200, lr=1.0)
    assert torch.equal(m[0].weight[0], m[0].weight[1])


def test_identical_nonzero_hidden_units_stay_identical_but_cannot_solve_xor():
    m = make_mlp("tanh", seed=3)
    with torch.no_grad():
        m[0].weight[1] = m[0].weight[0]; m[0].bias[1] = m[0].bias[0]
        m[2].weight[0, 1] = m[2].weight[0, 0]
    before = m[0].weight.clone()
    train(m, X, Y_BIN, bce, steps=2000, lr=1.0)
    assert not torch.equal(m[0].weight, before)             # it did learn something...
    assert torch.equal(m[0].weight[0], m[0].weight[1])      # ...but symmetry never broke
    assert bce(m(X), Y_BIN).item() > 0.3                    # and XOR is not solved


def test_softmax_sums_to_one_and_is_shift_invariant():
    z = torch.tensor([-2.6, 5.2, -2.8], dtype=torch.float64)
    p = torch.softmax(z, 0)
    assert abs(p.sum().item() - 1.0) < 1e-12
    assert torch.allclose(p, torch.softmax(z + 100.0, 0))


def test_ce_logit_gradient_is_p_minus_y():
    torch.manual_seed(0)
    z = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    nn.CrossEntropyLoss(reduction="sum")(z, Y_CLS).backward()
    y = torch.nn.functional.one_hot(Y_CLS, 3).double()
    assert torch.allclose(z.grad, torch.softmax(z.detach(), 1) - y)


def test_three_class_model_shapes_and_learning():
    m = make_mlp("tanh", n_hidden=2, n_out=3, seed=0)
    assert m[2].weight.shape == (3, 2) and m(X).shape == (4, 3)
    train(m, X, Y_CLS, nn.CrossEntropyLoss(), steps=5000, lr=0.5)
    assert torch.equal(m(X).argmax(1), Y_CLS)
