import torch
import pytest
from bitnet.model import activation_quant, weight_quant

@pytest.mark.parametrize("B,D", [(1,16), (8,128), (32,256)])
def test_act_no_amplification(B, D):
    x = torch.randn(B, D) * 5.0
    q = activation_quant(x)
    # each row’s peak abs after quant ≤ before quant
    before = x.abs().max(dim=-1).values
    after  = q.abs().max(dim=-1).values
    assert torch.all(after <= before + 1e-5)

def test_act_idempotent():
    x = torch.randn(4, 64)
    q1 = activation_quant(x)
    q2 = activation_quant(q1)
    assert torch.allclose(q1, q2, atol=1e-6)

def test_act_zero():
    z = torch.zeros(3, 128)
    assert torch.all(activation_quant(z) == 0)

def test_activation_quant_max_level():
    x = torch.zeros(1, 16)
    x[0, 0] = 1e6
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-5)
    q = activation_quant(x)
    levels = (q * scale).round().clamp(-128, 127)
    assert levels[0, 0] == 127

def test_activation_quant_gradflow():
    x = torch.randn(2,3, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(activation_quant, (x,), atol=1e-4)

# ——— weight_quant ———

def test_weight_no_amplification():
    w = torch.randn(64, 32) * 2.0
    u = weight_quant(w)
    assert u.abs().max() <= w.abs().max() + 1e-6

def test_weight_zero():
    z = torch.zeros(10, 10)
    assert torch.all(weight_quant(z) == 0)

def test_weight_three_level_output():
    w = torch.randn(128, 64)
    u = weight_quant(w)
    # recompute scale inside test
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    levels = torch.unique((u * scale).round())
    # should only see -1, 0, or +1
    assert set(levels.tolist()).issubset({-1.0, 0.0, 1.0})
