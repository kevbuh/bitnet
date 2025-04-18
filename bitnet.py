import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, LayerNorm, Embedding

class CausalSelfAttention:
    def __init__(self, n_embd, n_head, block_size):
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.query = Linear(n_embd, n_embd, bias=False)
        self.key = Linear(n_embd, n_embd, bias=False)
        self.value = Linear(n_embd, n_embd, bias=False)
        self.proj = Linear(n_embd, n_embd, bias=False)
        # causal mask to ensure each position only attends to previous positions
        mask = np.tril(np.ones((block_size, block_size), dtype=np.float32))
        self.mask = Tensor(mask.reshape(1, 1, block_size, block_size))

    def __call__(self, x):
        B, T, C = x.shape
        # project to queries, keys, values
        q = self.query(x).reshape(B, T, self.n_head, self.head_dim).transpose(0,2,1,3)  # (B, nh, T, hd)
        k = self.key(x).reshape(B, T, self.n_head, self.head_dim).transpose(0,2,1,3)    # (B, nh, T, hd)
        v = self.value(x).reshape(B, T, self.n_head, self.head_dim).transpose(0,2,1,3)  # (B, nh, T, hd)
        # scaled dot-product attention
        att = q @ k.transpose(0,1,3,2)  # (B, nh, T, T)
        att = att * (1.0 / np.sqrt(self.head_dim))
        # apply causal mask
        causal_mask = self.mask[:, :, :T, :T]
        att = att.masked_fill(causal_mask == 0, float('-inf'))
        # softmax
        att = att.softmax(axis=-1)
        # attend to values
        y = att @ v  # (B, nh, T, hd)
        # reassemble heads
        y = y.transpose(0,2,1,3).reshape(B, T, C)
        # output projection
        return self.proj(y)

class FeedForward:
    def __init__(self, n_embd, expansion_factor=4):
        self.fc1 = Linear(n_embd, n_embd * expansion_factor)
        self.fc2 = Linear(n_embd * expansion_factor, n_embd)

    def __call__(self, x):
        return self.fc2(self.fc1(x).relu())

class Block:
    def __init__(self, n_embd, n_head, block_size):
        self.ln1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln2 = LayerNorm(n_embd)
        self.ff = FeedForward(n_embd)

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))  # residual
        x = x + self.ff(self.ln2(x))    # residual
        return x

class GPT:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        self.vocab_size = vocab_size
        self.token_emb = Embedding(vocab_size, n_embd)
        # learned positional embeddings
        pos = np.random.randn(1, block_size, n_embd).astype(np.float32) * 0.01
        self.pos_emb = Tensor(pos)
        # transformer blocks
        self.blocks = [Block(n_embd, n_head, block_size) for _ in range(n_layer)]
        self.ln_f = LayerNorm(n_embd)
        # language modeling head
        self.head = Linear(n_embd, vocab_size, bias=False)
        self.block_size = block_size

    def __call__(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds block size"
        # token embeddings + positional embeddings
        tok_emb = self.token_emb(idx)        # (B, T, C)
        pos_emb = self.pos_emb[:, :T, :]     # (1, T, C)
        x = tok_emb + pos_emb
        # forward through transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # output logits
        logits = self.head(x)                # (B, T, vocab_size)
        return logits

# Example usage:
# model = GPT(vocab_size=50257, n_embd=768, n_head=12, n_layer=12, block_size=1024)
# idx = Tensor(np.random.randint(0, 50257, (1, 16), dtype=np.int32))
# logits = model(idx)
