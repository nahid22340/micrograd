import torch
from engine import value

def test_sanity_check():
    """Verifies basic arithmetic and autograd against PyTorch."""
    # Custom Value Engine
    x1 = value(2.0, label='x1')
    x2 = value(0.0, label='x2')
    w1 = value(-3.0, label='w1')
    w2 = value(1.0, label='w2')
    b = value(6.881373587019543, label='b')
    x1w1 = x1 * w1
    x2w2 = x2 * w2
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b
    o = n.tanh()
    o.backward()

    # PyTorch Verification
    x1_t = torch.tensor([2.0], requires_grad=True, dtype=torch.float64)
    x2_t = torch.tensor([0.0], requires_grad=True, dtype=torch.float64)
    w1_t = torch.tensor([-3.0], requires_grad=True, dtype=torch.float64)
    w2_t = torch.tensor([1.0], requires_grad=True, dtype=torch.float64)
    b_t = torch.tensor([6.881373587019543], requires_grad=True, dtype=torch.float64)
    n_t = x1_t * w1_t + x2_t * w2_t + b_t
    o_t = torch.tanh(n_t)
    o_t.backward()

    # Assert outputs and gradients match to 5 decimal places
    assert abs(o.data - o_t.item()) < 1e-5
    assert abs(x1.grad - x1_t.grad.item()) < 1e-5
    assert abs(w1.grad - w1_t.grad.item()) < 1e-5

def test_gradient_accumulation():
    """Verifies multi-use variable gradient accumulation (+=)."""
    a = value(3.0)
    b = a + a  # b = 2a -> db/da = 2
    b.backward()
    assert a.grad == 2.0