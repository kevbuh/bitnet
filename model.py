import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.linen import initializers
from typing import Optional, Callable


def quantize_bits(x: jnp.ndarray, bits: int) -> jnp.ndarray:
    """
    Quantize tensor to fixed-point representation with given bits.
    """
    qmin = 0
    qmax = 2**bits - 1
    # map to [0,1]
    x_min = x.min()
    x_max = x.max()
    # avoid div0
    scale = (x_max - x_min) / (qmax - qmin + 1e-8)
    x_norm = (x - x_min) / (scale + 1e-8)
    x_q = jnp.round(jnp.clip(x_norm, qmin, qmax))
    x_deq = x_q * scale + x_min
    return x_deq


class BitDense(nn.Module):
    """
    Linear layer with weight quantized to a fixed number of bits.
    No bias term.
    """
    features: int
    bits: int = 8
    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32
    kernel_init: Callable = initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        input_features = inputs.shape[-1]
        w = self.param('kernel', self.kernel_init, (input_features, self.features))
        # quantize weights
        w_q = quantize_bits(w, self.bits)
        y = jnp.dot(inputs, w_q.astype(self.dtype))
        return y


def squared_relu(x: jnp.ndarray) -> jnp.ndarray:
    """ReLU squared activation: (max(0, x))^2"""
    return jnp.square(jnp.clip(x, a_min=0.0))


class SubLayerNorm(nn.Module):
    """
    Sublayer layer normalization (Pre-LN), no bias.
    """
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return nn.LayerNorm(use_bias=False, use_scale=True, epsilon=self.epsilon)(x)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)"""
    dim: int

    @nn.compact
    def __call__(self, seq_len: int, dtype=jnp.float32):
        inv_freq = 1.0 / (10000 ** (jnp.arange(0, self.dim, 2) / self.dim))
        t = jnp.arange(seq_len, dtype=dtype)
        freqs = jnp.einsum('i , j -> i j', t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        sin, cos = jnp.sin(emb), jnp.cos(emb)
        return sin, cos


def apply_rotary(x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray) -> jnp.ndarray:
    """Apply rotary embedding to tensor x"""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


class MultiHeadSelfAttention(nn.Module):
    embed_dim: int
    num_heads: int
    bits: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True):
        head_dim = self.embed_dim // self.num_heads
        assert head_dim * self.num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        batch, seq_len, _ = x.shape

        # project inputs
        qkv_proj = BitDense(self.embed_dim * 3, bits=self.bits, use_bias=False)(x)
        qkv = qkv_proj.reshape(batch, seq_len, 3, self.num_heads, head_dim)
        q, k, v = jnp.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)

        # rotary embeddings
        sin, cos = RotaryEmbedding(head_dim)(seq_len)
        q = apply_rotary(q, sin, cos)
        k = apply_rotary(k, sin, cos)

        # scaled dot-product
        attn_scores = jnp.einsum('bhqd, bhkd -> bhqk', q, k) / jnp.sqrt(head_dim)
        attn_weights = nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('bhqk, bhvd -> bhqd', attn_weights, v)
        attn_out = attn_out.reshape(batch, seq_len, self.embed_dim)

        # output projection
        out = BitDense(self.embed_dim, bits=self.bits, use_bias=False)(attn_out)
        return out


class FeedForward(nn.Module):
    embed_dim: int
    hidden_dim: int
    bits: int = 8

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # first linear
        h = BitDense(self.hidden_dim, bits=self.bits, use_bias=False)(x)
        # activation
        h = squared_relu(h)
        # second linear
        out = BitDense(self.embed_dim, bits=self.bits, use_bias=False)(h)
        return out


class TransformerBlock(nn.Module):
    embed_dim: int
    num_heads: int
    hidden_dim: int
    bits: int = 8
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool = True):
        # Self-attention
        ln1 = SubLayerNorm()(x)
        attn = MultiHeadSelfAttention(self.embed_dim, self.num_heads, bits=self.bits)(ln1)
        attn = nn.Dropout(rate=self.dropout_rate)(attn, deterministic)
        x = x + attn

        # Feed-forward
        ln2 = SubLayerNorm()(x)
        ff = FeedForward(self.embed_dim, self.hidden_dim, bits=self.bits)(ln2)
        ff = nn.Dropout(rate=self.dropout_rate)(ff, deterministic)
        x = x + ff
        return x


class Transformer(nn.Module):
    vocab_size: int
    max_seq_len: int
    num_layers: int
    embed_dim: int
    num_heads: int
    hidden_dim: int
    bits: int = 8
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, *, deterministic: bool = True):
        # Embedding
        embed_init = initializers.normal(stddev=1.0)
        token_emb = nn.Embed(self.vocab_size, self.embed_dim, embedding_init=embed_init, name='token_embed', dtype=jnp.float32)
        x = token_emb(input_ids)

        # Position embeddings via RoPE inside attention

        for _ in range(self.num_layers):
            x = TransformerBlock(self.embed_dim, self.num_heads, self.hidden_dim, bits=self.bits, dropout_rate=self.dropout_rate)(x, deterministic=deterministic)

        x = SubLayerNorm()(x)
        logits = BitDense(self.vocab_size, bits=self.bits, use_bias=False)(x)
        return logits
