"""
Cleaned BitNet b1.58 2B‑4T JAX/Flax loader and inference script
"""

import json
import numpy as np
from safetensors import safe_open

import jax
import jax.numpy as jnp
from flax import linen as nn
from transformers import GPT2TokenizerFast


def load_weight_groups(path: str) -> dict:
    """
    Load safetensors file and group weights by layer or top‑level name.

    Returns:
        groups: dict where
          - layers: groups['layer.0']['<block>']['<param_name>'] holds suffix dicts
          - top‑level: e.g. groups['embed_tokens'], groups['norm']
    """
    groups = {}
    with safe_open(path, framework="np") as f:
        for full_key in f.keys():
            key_path, suffix = full_key.rsplit('.', 1)
            if key_path.startswith("model."):
                key_path = key_path[len("model."):]
            parts = key_path.split('.')

            if parts[0] == 'layers':
                _, idx, block, *name_parts = parts
                bucket = groups.setdefault(f'layer.{idx}', {})
                bucket = bucket.setdefault(block, {})
                bucket = bucket.setdefault('.'.join(name_parts), {})
            else:
                bucket = groups.setdefault(parts[0], {})

            if suffix in ('weight', 'weight_sign', 'weight_scale'):
                bucket[suffix] = f.get_tensor(full_key)
    return groups


def decode_bit_weights(sign: np.ndarray, scale: np.ndarray) -> jnp.ndarray:
    """
    Decode quantized (sign, scale) arrays to float32 weight matrix.
    """
    return jnp.array(sign, jnp.float32) * jnp.array(scale, jnp.float32)[..., None]


def build_flax_params(groups: dict) -> dict:
    """
    Convert grouped weights into a flat dict of JAX arrays for Flax.
    """
    params = {}

    # Transformer layers
    for layer_key, blocks in groups.items():
        if not layer_key.startswith('layer.'):
            continue
        params[layer_key] = {'attn': {}, 'ffn': {}}

        # Attention block maps
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            bucket = blocks.get('self_attn', blocks.get('attn', {})).get(name, {})
            if 'weight' in bucket:
                w = jnp.array(bucket['weight'])
            else:
                w = decode_bit_weights(bucket['weight_sign'], bucket['weight_scale'])
            params[layer_key]['attn'][name] = w.reshape(-1, w.shape[-1])

        # Feed-forward block maps
        mlp_block = blocks.get('mlp', {})
        # gate_proj -> wi, up_proj -> wo
        if 'gate_proj' in mlp_block and 'up_proj' in mlp_block:
            for alias, key in (('wi', 'gate_proj'), ('wo', 'up_proj')):
                bucket = mlp_block[key]
                if 'weight' in bucket:
                    w = jnp.array(bucket['weight'])
                else:
                    w = decode_bit_weights(bucket['weight_sign'], bucket['weight_scale'])
                params[layer_key]['ffn'][alias] = w.reshape(-1, w.shape[-1])

    # Top-level: embeddings and norms
    for key, bucket in groups.items():
        if key.startswith('layer.'):
            continue
        if 'weight' in bucket:
            params[key] = jnp.array(bucket['weight'])
        else:
            params[key] = decode_bit_weights(bucket['weight_sign'], bucket['weight_scale'])

    return params


class BitLinear(nn.Module):
    weight: jnp.ndarray

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x @ self.weight.T


class BitSelfAttention(nn.Module):
    cfg: dict
    weights: dict  # q_proj, k_proj, v_proj, o_proj

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, t, _ = x.shape
        # Project inputs
        q_out = x @ self.weights['q_proj'].T  # [b, t, out_q]
        k_out = x @ self.weights['k_proj'].T  # [b, t, out_k]
        v_out = x @ self.weights['v_proj'].T  # [b, t, out_v]
        # Head configurations
        h_q = self.cfg['num_attention_heads']
        h_kv = self.cfg['num_key_value_heads']
        # Compute per-head dimensions
        dh_q = q_out.shape[-1] // h_q
        dh_k = k_out.shape[-1] // h_kv
        # Reshape and transpose to [b, h, t, d]
        q = q_out.reshape(b, t, h_q, dh_q).transpose(0,2,1,3)
        k = k_out.reshape(b, t, h_kv, dh_k).transpose(0,2,1,3)
        v = v_out.reshape(b, t, h_kv, dh_k).transpose(0,2,1,3)
        # Broadcast K and V to match attention heads
        group_size = h_q // h_kv
        k = jnp.repeat(k, group_size, axis=1)
        v = jnp.repeat(v, group_size, axis=1)
        # Attention scores
        attn_logits = (q @ k.transpose(0,1,3,2)) / jnp.sqrt(dh_q)
        mask = jnp.tril(jnp.ones((t, t)))
        attn = nn.softmax(jnp.where(mask, attn_logits, -1e10), axis=-1)
        # Context
        ctx = (attn @ v).transpose(0,2,1,3).reshape(b, t, self.weights['o_proj'].shape[0])
        # Output projection
        out = ctx @ self.weights['o_proj']
        return out

class BitFFN(nn.Module):
    weights: dict  # wi, wo

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hid = x @ self.weights['wi'].T
        hid = jnp.square(nn.relu(hid))
        return hid @ self.weights['wo']


class BitDecoderLayer(nn.Module):
    cfg: dict
    params: dict

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_norm = nn.LayerNorm()(x)
        attn_out = BitSelfAttention(self.cfg, self.params['attn'])(x_norm)
        x = x + attn_out

        x_norm = nn.LayerNorm()(x)
        ffn_out = BitFFN(self.params['ffn'])(x_norm)
        return x + ffn_out


class BitNetModel(nn.Module):
    cfg: dict
    params: dict

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        # Use pre‑loaded embedding matrix
        embed_tokens = self.params['embed_tokens']  # [vocab, hidden]
        x = jnp.take(embed_tokens, input_ids, axis=0)

        pos = self.param('pos_embed', nn.initializers.normal(0.02),
                          (1, self.cfg['max_position_embeddings'], self.cfg['hidden_size']))
        x = x + pos[:, :x.shape[1], :]

        for i in range(self.cfg['num_hidden_layers']):
            layer_key = f'layer.{i}'
            x = BitDecoderLayer(self.cfg, self.params[layer_key])(x)

        x = nn.LayerNorm()(x)
        # tied LM head
        logits = x @ embed_tokens.T
        return logits


if __name__ == '__main__':
    cfg = json.load(open('bitnet-b1.58-2B-4T/config.json'))
    groups = load_weight_groups('bitnet-b1.58-2B-4T/model.safetensors')
    print('Top-level keys:', [k for k in groups if not k.startswith('layer.')])
    params = build_flax_params(groups)

    model = BitNetModel(cfg, params)
    rng = jax.random.PRNGKey(0)
    dummy = jnp.ones((1,1), jnp.int32)
    vars = model.init(rng, dummy)

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    @jax.jit
    def step(ids):
        return model.apply(vars, ids)

    def generate(prompt: str, max_new: int=50) -> str:
        ids = tokenizer(prompt, return_tensors='jax')['input_ids']
        for _ in range(max_new):
            print("*")
            logits = step(ids)
            nxt = jnp.argmax(logits[0, -1])
            ids = jnp.concatenate([ids, nxt[None, None]], axis=1)
        return tokenizer.decode(ids[0])

    print(generate('Hello, world', max_new=20))
