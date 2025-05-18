# https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from BitLinear import BitLinear
from utils import print_model_params, training_step, print_weights, timeit
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

DEBUG = True

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(cached_batches=None):
    out = {}
    model.eval()
    
    # Use cached batches if provided, otherwise create new ones
    if cached_batches is None:
        cached_batches = {}
        for split in ['train', 'val']: cached_batches[split] = [get_batch(split) for _ in range(eval_iters)]
    
    for split in ['train', 'val']:
        losses = torch.zeros(min(eval_iters, len(cached_batches[split])))
        for k, (X, Y) in enumerate(cached_batches[split]):
            if k >= eval_iters: break
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out, cached_batches

# class Head(nn.Module):
#     """ one head of self-attention """

#     def __init__(self, head_size):
#         super().__init__()
#         self.key = BitLinear(n_embd, head_size, bias=False)
#         self.query = BitLinear(n_embd, head_size, bias=False)
#         self.value = BitLinear(n_embd, head_size, bias=False)
#         self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         # input of size (batch, time-step, channels)
#         # output of size (batch, time-step, head size)
#         B,T,C = x.shape
#         k = self.key(x)   # (B,T,hs)
#         q = self.query(x) # (B,T,hs)
#         # compute attention scores ("affinities")
#         wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
#         wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
#         wei = F.softmax(wei, dim=-1) # (B, T, T)
#         wei = self.dropout(wei)
#         # perform the weighted aggregation of the values
#         v = self.value(x) # (B,T,hs)
#         out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
#         return out

# class MultiHeadAttention(nn.Module):
#     """ multiple heads of self-attention in parallel """

#     def __init__(self, num_heads, head_size):
#         super().__init__()
#         self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
#         self.proj = BitLinear(head_size * num_heads, n_embd)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         out = torch.cat([h(x) for h in self.heads], dim=-1)
#         out = self.dropout(self.proj(out))
#         return out
    
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
        self.dropout = nn.Dropout(dropout)

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

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, hidden_dim):
        super().__init__()
        self.gate_proj = BitLinear(d_model, hidden_dim, bias=False)
        self.up_proj   = BitLinear(d_model, hidden_dim, bias=False)
        self.down_proj = BitLinear(hidden_dim, d_model, bias=False)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
    
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).rsqrt()
        return self.weight * x * norm

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_ln   = RMSNorm(n_embd)
        self.attn       = RotaryMHA(n_embd, n_head, n_kv_head)
        self.post_ln    = RMSNorm(n_embd)
        self.mlp        = SwiGLUFFN(n_embd, ffn_dim)
    def forward(self, x):
        x = x + self.attn(self.input_ln(x))
        x = x + self.mlp(self.post_ln(x))
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, n_embd)
        self.pos_embed    = nn.Parameter(torch.zeros(block_size, n_embd))  # learned rope alt.
        self.layers       = nn.ModuleList([Block() for _ in range(n_layer)])
        self.ln_f         = RMSNorm(n_embd)
        self.lm_head      = BitLinear(n_embd, vocab_size, bias=False)

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

    @timeit(debug=DEBUG)
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop idx to the last block_size tokens
            logits, loss = self(idx_cond) # get the predictions
            logits = logits[:, -1, :] # becomes (B, C), focus only on the last time step
            probs = F.softmax(logits, dim=-1) # (B, C), apply softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1), sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1), append sampled index to the running sequence
        return idx
    
    @timeit(debug=DEBUG)
    def stream_output(self, max_new_tokens):
        model.eval()
        idx = torch.zeros((1, 1), device=device, dtype=torch.long) # initialize context
        chars = sorted(list(set(text))) # build your id→char map
        itos = { i: ch for i, ch in enumerate(chars) }
        print("decoding…", flush=True)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop to last block_size tokens
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # (1, vocab_size)
            probs = F.softmax(logits, dim=-1) # (1, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1) # (1,1)
            idx = torch.cat([idx, idx_next], dim=1)
            print(itos[idx_next.item()], end='', flush=True)
        print()

def calculate_model_size_in_gb(model):
    total_params = sum(p.numel() for p in model.parameters())
    # Assuming float32, which is 4 bytes per parameter
    total_size_bytes = total_params * 4
    total_size_gb = total_size_bytes / (1024 ** 3)
    print(f"Model size: {total_size_gb:.2f} GB")
    return total_size_gb

if __name__ == "__main__":
    # ------------
    # hyperparameters
    batch_size = 1 # how many independent sequences will we process in parallel?
    max_iters = 100
    eval_interval = 500
    eval_iters = 50
    learning_rate = 3e-4
    # device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    print(f"Using device: {device}")
    # ------------
    # archparameters
    n_embd      = 2560          # hidden size d
    n_head      = 32            # total attention heads
    n_kv_head   = 8             # GQA: heads that carry K & V
    head_dim    = n_embd // n_head # 80
    ffn_dim     = 6912
    n_layer     = 30
    vocab_size  = 128_256
    block_size  = 2048          # anything ≥ max context used in training
    dropout     = 0.0           # llama uses no dropout after pre-training
    cached_batches = None
    # ------------
    model = GPTLanguageModel()
    calculate_model_size_in_gb(model)
    m = model.to(device).half()
    if torch.cuda.is_available(): m = torch.compile(m)
    print_model_params(m)
    if torch.cuda.is_available(): training_step = torch.compile(training_step)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # ------------
    print(f"Training for {max_iters} iterations")
    for iter in tqdm(range(max_iters)):
        xb, yb = get_batch('train')
        # if DEBUG:
        #     if iter % eval_interval == 0 or iter == max_iters - 1:
        #         # loss_dict, cached_batches = estimate_loss(cached_batches)
        #         with torch.no_grad():
        #             logits, loss = model(xb, yb)
        #             print(f"step {iter}: train loss {loss:.4f}")
        training_step(model, xb, yb, optimizer)
    if DEBUG: print_weights(m)
    m.stream_output(block_size)