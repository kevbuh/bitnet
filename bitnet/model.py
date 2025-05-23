import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ---- quant implementations from the paper: https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

def activation_quant(x):
  """ Per-token quantization to 8 bits. No grouping is needed for quantization.
  Args: x: an activation tensor with shape [n, d]
  Returns: y: a quantized activation tensor with shape [n, d]
  """
  scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
  y = (x * scale).round().clamp_(-128, 127) / scale
  return y

def weight_quant(w):
  """ Per-tensor quantization to 1.58 bits. No grouping is needed for quantization.
  Args: w: a weight tensor with shape [d, k]
  Returns: u: a quantized weight with shape [d, k]
  """
  scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
  u = (w * scale).round().clamp_(-1, 1) / scale
  return u
    
class BitLinear(nn.Module):
  """
  This is only for training, and kernel optimization is needed for efficiency.
  """
  def __init__(self, in_features, out_features):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = nn.Parameter(torch.empty((out_features, in_features)))
    self.norm = nn.RMSNorm(in_features)
    nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
      
  def forward(self, x):
    """
    Args: x: an input tensor with shape [n, d]
    Returns: y: an output tensor with shape [n, d]
    """
    # NOTE: the paper says not to use normalization, but then contradicts it by using RMSNorm in their code. dont use for now?
    # x_quant = x
    x_norm = self.norm(x)
    x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach() # A trick for implementing Straight−Through−Estimator (STE) using detach()
    w_quant = self.weight + (weight_quant(self.weight) - self.weight).detach()
    y = F.linear(x_quant, w_quant)
    return y
  
# ------- custom bitnet layers -------

class RotaryMHA(nn.Module):
  def __init__(self, d_model, n_head, n_kv_head, block_size):
    super().__init__()
    self.n_head = n_head
    self.n_kv   = n_kv_head
    self.head_dim = d_model // n_head

    inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2) / self.head_dim))
    self.register_buffer("inv_freq", inv_freq)

    # precompute 1×1×block_size×block_size causal mask once
    mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
    self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(1))

    self.q_proj = BitLinear(d_model, d_model)                    # 2560×2560
    self.k_proj = BitLinear(d_model, self.head_dim * n_kv_head)  # 2560×640
    self.v_proj = BitLinear(d_model, self.head_dim * n_kv_head)  # 2560×640
    self.o_proj = BitLinear(d_model, d_model)                    # 2560×2560

  def forward(self, x):
    B, T, _ = x.shape
    q = self.q_proj(x).view(B, T, self.n_head,  self.head_dim)  # B T 32 80
    k = self.k_proj(x).view(B, T, self.n_kv,   self.head_dim)   # B T  8 80
    v = self.v_proj(x).view(B, T, self.n_kv,   self.head_dim)

    # broadcast K-V to all heads (Llama GQA)
    k = k.repeat_interleave(self.n_head // self.n_kv, dim=2)
    v = v.repeat_interleave(self.n_head // self.n_kv, dim=2)

    # ── RoPE ──────────────────────────────────────────────
    seq = torch.arange(T, device=x.device)
    freqs = torch.einsum("t , d -> t d", seq, self.inv_freq)     # T × (d/2)
    cos, sin = freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :]

    def rope(t):
      t1, t2 = t[..., ::2], t[..., 1::2]
      return torch.cat([t1 * cos - t2 * sin, t1 * sin + t2 * cos], dim=-1)

    q, k = rope(q), rope(k)
    # ─────────────────────────────────────────────────────

    q = q.permute(0, 2, 1, 3)    # B, 32, T, 80
    k = k.permute(0, 2, 1, 3)    # B,  8, T, 80  (then repeat_interleave)
    v = v.permute(0, 2, 1, 3)

    attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5   # B, 32, T, T
    # grab the top-left T×T of our prebuilt mask
    mask = self.causal_mask[:, :, :T, :T]
    attn  = attn.masked_fill(~mask, float('-inf'))
    attn  = attn.softmax(-1)
    out   = (attn @ v).transpose(1, 2).reshape(B, T, -1)       # back to B, T, 2560
    return self.o_proj(out)

class ReLUSq(nn.Module):
  def forward(self, x):
    return F.relu(x) ** 2

class ReLUSqFFN(nn.Module):
  def __init__(self, d_model, hidden_dim):
    super().__init__()
    self.fc1 = BitLinear(d_model, hidden_dim)
    self.act = ReLUSq()
    self.fc2 = BitLinear(hidden_dim, d_model)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)
    return x

class SubLayerNorm(nn.Module):
  def __init__(self, dim, eps=1e-6):
    super().__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    return (x-x.mean(-1, keepdim=True)) * self.gamma
    
class Block(nn.Module):
  def __init__(self, n_embd, n_head, n_kv_head, ffn_dim, block_size):
    super().__init__()
    self.input_ln   = SubLayerNorm(n_embd)
    self.attn       = RotaryMHA(n_embd, n_head, n_kv_head, block_size)
    self.post_ln    = SubLayerNorm(n_embd)
    self.mlp        = ReLUSqFFN(n_embd, ffn_dim)

  def forward(self, x):
    x = x + self.attn(self.input_ln(x))
    x = x + self.mlp(self.post_ln(x))
    return x

class BitNet(nn.Module):
  def __init__(self, vocab_size, d_model, block_size, n_layer, n_head, n_kv_head, ffn_dim):
    if n_head % n_kv_head != 0: raise ValueError("n_head must be divisible by n_kv_head")
    super().__init__()
    self.block_size = block_size
    self.embed_tokens = nn.Embedding(vocab_size, d_model)
    self.layers       = nn.ModuleList([Block(d_model, n_head, n_kv_head, ffn_dim, block_size) for _ in range(n_layer)])
    self.ln_f         = SubLayerNorm(d_model)
    self.lm_head      = BitLinear(d_model, vocab_size)

  def forward(self, idx, targets=None):
    x = self.embed_tokens(idx)
    for layer in self.layers: x = layer(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B,T,vocab_size)

    if targets is None: loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  def generate(self, max_new_tokens, tokenizer, device):
    self.eval()
    idx = torch.zeros((1, 1), device=device, dtype=torch.long) # initialize context
    print("decoding…", flush=True)
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size:] # crop to last block_size tokens
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :] # (1, vocab_size)
      probs = F.softmax(logits, dim=-1) # (1, vocab_size)
      idx_next = torch.multinomial(probs, num_samples=1) # (1,1)
      idx = torch.cat([idx, idx_next], dim=1)
      print(tokenizer.decode([idx_next.item()], skip_special_tokens=True), end='', flush=True)
    print()
