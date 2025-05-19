import torch
import torch.nn as nn
from torch.nn import functional as F

from BitLinear import BitLinear
from utils import timeit

class RotaryMHA(nn.Module):
  def __init__(self, d_model, n_head, n_kv_head):
    super().__init__()
    self.n_head = n_head
    self.n_kv   = n_kv_head
    self.head_dim = d_model // n_head
    self.q_proj = BitLinear(d_model, d_model, bias=False)               # 2560×2560
    self.k_proj = BitLinear(d_model, self.head_dim * n_kv_head, False)  # 2560×640
    self.v_proj = BitLinear(d_model, self.head_dim * n_kv_head, False)  # 2560×640
    self.o_proj = BitLinear(d_model, d_model, bias=False)               # 2560×2560

  def forward(self, x):
    B, T, _ = x.shape
    q = self.q_proj(x).view(B, T, self.n_head,  self.head_dim)  # B T 32 80
    k = self.k_proj(x).view(B, T, self.n_kv,   self.head_dim)   # B T  8 80
    v = self.v_proj(x).view(B, T, self.n_kv,   self.head_dim)

    # broadcast K-V to all heads (Llama GQA)
    k = k.repeat_interleave(self.n_head // self.n_kv, dim=2)
    v = v.repeat_interleave(self.n_head // self.n_kv, dim=2)

    # rotary-pos-enc here if you like …

    q = q.permute(0, 2, 1, 3)    # B, 32, T, 80
    k = k.permute(0, 2, 1, 3)    # B,  8, T, 80  (then repeat_interleave)
    v = v.permute(0, 2, 1, 3)

    attn = (q @ k.transpose(-2, -1)) * self.head_dim ** -0.5   # B, 32, T, T
    mask  = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
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
    self.fc1 = BitLinear(d_model, hidden_dim, bias=False)
    self.act = ReLUSq()
    self.fc2 = BitLinear(hidden_dim, d_model, bias=False)

  def forward(self, x):
    x = self.fc1(x)
    x = self.act(x)
    x = self.fc2(x)
    return x

class SubLayerNorm(nn.Module):
  def __init__(self, dim, eps=1e-6, elementwise_affine=True):
    super().__init__()
    self.eps = eps
    if elementwise_affine:
      self.gamma = nn.Parameter(torch.ones(dim))
      self.beta  = nn.Parameter(torch.zeros(dim))
    else:
      self.register_parameter('gamma', None)
      self.register_parameter('beta',  None)

  def forward(self, x):
    mean = x.mean(-1, keepdim=True) # Compute mean over last dimension
    var = (x - mean).pow(2).mean(-1, keepdim=True) # Compute variance over last dimension
    x_norm = (x - mean) / torch.sqrt(var + self.eps) # Normalize: zero-mean, unit-variance
    if self.gamma is not None: x_norm = x_norm * self.gamma + self.beta # Apply affine if present
    return x_norm
    
class Block(nn.Module):
  def __init__(self, n_embd, n_head, n_kv_head, ffn_dim):
    super().__init__()
    self.input_ln   = SubLayerNorm(n_embd)
    self.attn       = RotaryMHA(n_embd, n_head, n_kv_head)
    self.post_ln    = SubLayerNorm(n_embd)
    self.mlp        = ReLUSqFFN(n_embd, ffn_dim)
  def forward(self, x):
    x = x + self.attn(self.input_ln(x))
    x = x + self.mlp(self.post_ln(x))
    return x

class GPTLanguageModel(nn.Module):
  def __init__(self, vocab_size, d_model, block_size, n_layer, n_head, n_kv_head, ffn_dim):
    super().__init__()
    self.block_size = block_size
    self.embed_tokens = nn.Embedding(vocab_size, d_model)
    self.pos_embed    = nn.Parameter(torch.zeros(block_size, d_model))  # learned rope alt.
    self.layers       = nn.ModuleList([Block(d_model, n_head, n_kv_head, ffn_dim) for _ in range(n_layer)])
    self.ln_f         = SubLayerNorm(d_model)
    self.lm_head      = BitLinear(d_model, vocab_size, bias=False)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, BitLinear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None: torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding): torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, idx, targets=None):
    x = self.embed_tokens(idx) + self.pos_embed[:idx.size(1)]
    for layer in self.layers:
      x = layer(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) # (B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  @timeit()
  def generate(self, idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -self.block_size:] # crop idx to the last block_size tokens
      logits, loss = self(idx_cond) # get the predictions
      logits = logits[:, -1, :] # becomes (B, C), focus only on the last time step
      probs = F.softmax(logits, dim=-1) # (B, C), apply softmax to get probabilities
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1), sample from the distribution
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1), append sampled index to the running sequence
    return idx
  
  @timeit()
  def stream_output(self, max_new_tokens, itos, device):
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
      print(itos[idx_next.item()], end='', flush=True)
    print()