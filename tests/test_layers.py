import torch
from bitnet.model import SubLayerNorm, ReLUSq, ReLUSqFFN, BitLinear, RotaryMHA

def test_sublayernorm_zero_mean():
    B, D = 3, 5
    x = torch.randn(B, D)
    norm = SubLayerNorm(D)
    y = norm(x)
    means = y.mean(dim=-1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-6)

def test_sublayernorm_gamma_scaling():
    B, D = 2, 4
    x = torch.randn(B, D)
    norm = SubLayerNorm(D)
    norm.gamma.data.fill_(3.0)
    y = norm(x)
    centered = x - x.mean(dim=-1, keepdim=True)
    assert torch.allclose(y, centered * 3.0, atol=1e-6)

def test_relusq_behavior():
    act = ReLUSq()
    x = torch.tensor([[-1.0, 0.5, 2.0]])
    y = act(x)
    expected = torch.tensor([[0.0, 0.5**2, 2.0**2]])
    assert torch.allclose(y, expected, atol=1e-6)

def test_relusqffn_output_shape():
    B, T, D, H = 2, 3, 8, 16
    ffn = ReLUSqFFN(d_model=D, hidden_dim=H)
    x = torch.randn(B, T, D)
    y = ffn(x)
    assert y.shape == (B, T, D)

def test_bitlinear_shape_and_dtype():
    B, D_in, D_out = 4, 7, 5
    layer = BitLinear(D_in, D_out)
    x = torch.randn(B, D_in)
    y = layer(x)
    assert y.shape == (B, D_out)
    assert y.dtype == x.dtype

def test_rotarymha_shape_and_mask():
    B, T, D = 2, 6, 8
    n_head, n_kv = 2, 1
    mha = RotaryMHA(d_model=D, n_head=n_head, n_kv_head=n_kv, block_size=10)
    x = torch.randn(B, T, D)
    out = mha(x)
    assert out.shape == x.shape
    mask = mha.causal_mask[0, 0, :T, :T]
    tril = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert torch.equal(mask, tril)

def test_rope_basic_rotation():
    # construct sin/cos matrices for a small head_dim and seq length
    head_dim = 4
    seq_len = 3
    B = 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2) / head_dim))
    seq = torch.arange(seq_len)
    freqs = torch.einsum("t,d->td", seq, inv_freq)
    cos = freqs.cos()[None, :, None, :]
    sin = freqs.sin()[None, :, None, :]

    # create a tensor where even dims are 1 and odd dims 0
    t = torch.zeros((B, seq_len, 1, head_dim))
    t[..., ::2] = 1.0

    # apply the static rope function
    result = RotaryMHA.rope(t, sin, cos)

    # manual expected rotation: [t1*cos - t2*sin, t1*sin + t2*cos]
    t1, t2 = t[..., ::2], t[..., 1::2]
    expected = torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)
    assert torch.allclose(result, expected, atol=1e-6)
