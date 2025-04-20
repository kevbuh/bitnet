import json
import numpy as np
from safetensors import safe_open

import jax
import jax.numpy as jnp
from flax import linen as nn

# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def load_weight_groups(path: str) -> dict:
    """Load safetensors file and group weights by layer or top‑level key."""
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
    """Decode ternary sign + per‑row scale → float32 weight."""
    return jnp.array(sign, jnp.float32) * jnp.array(scale, jnp.float32)[..., None]


# -------------------------------------------------------------
# Parameter loader
# -------------------------------------------------------------

def _maybe_transpose(w: jnp.ndarray, cfg: dict, proj_name: str) -> jnp.ndarray:
    """Ensure each projection weight has the expected orientation.

    The BitNet safetensors store q/k/v/o weights transposed compared to
    the usual HF layout.  Empirically:
      * q_proj / k_proj / v_proj  :  (qkv_dim, hidden)   (OK as‑is)
      * o_proj                    :  (hidden,  qkv_dim)  (NEEDS T)
      * wi (gate)                 :  (ffn_dim, hidden)   (OK)
      * wo (up)                   :  (hidden,  ffn_dim)  (NEEDS T)
    """
    h = cfg['hidden_size']
    if proj_name in ('o_proj', 'wo'):
        # want shape (hidden, in_dim)
        if w.shape[0] != h and w.shape[1] == h:
            w = w.T
    else:
        # want shape (out_dim, hidden)
        if w.shape[1] != h and w.shape[0] == h:
            w = w.T
    return w


def build_flax_params(groups: dict, cfg: dict) -> dict:
    """Decode / orient all weights, return bf16 params dict."""
    params = {}

    # Transformer layers ---------------------------------------------------
    for layer_key, blocks in groups.items():
        if not layer_key.startswith('layer.'):
            continue
        params[layer_key] = {'attn': {}, 'ffn': {}}

        # Attention projections --------------------------------------------
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            bucket = blocks.get('self_attn', blocks.get('attn', {})).get(name, {})
            if 'weight' in bucket:
                w = jnp.array(bucket['weight'])
            else:
                w = decode_bit_weights(bucket['weight_sign'], bucket['weight_scale'])
            w = _maybe_transpose(w, cfg, name).astype(jnp.bfloat16)
            params[layer_key]['attn'][name] = w

        # Feed‑forward projections -----------------------------------------
        mlp = blocks.get('mlp', {})
        for alias, key in (('wi', 'gate_proj'), ('wo', 'up_proj')):
            bucket = mlp.get(key, {})
            if 'weight' in bucket:
                w = jnp.array(bucket['weight'])
            else:
                w = decode_bit_weights(bucket['weight_sign'], bucket['weight_scale'])
            w = _maybe_transpose(w, cfg, alias).astype(jnp.bfloat16)
            params[layer_key]['ffn'][alias] = w

    # Top‑level embeddings & final norm ------------------------------------
    for key, bucket in groups.items():
        if key.startswith('layer.'):
            continue
        if 'weight' in bucket:
            w = jnp.array(bucket['weight'])
        else:
            w = decode_bit_weights(bucket['weight_sign'], bucket['weight_scale'])
        params[key] = w.astype(jnp.bfloat16)

    return params

# -------------------------------------------------------------
# Modules
# -------------------------------------------------------------
class BitSelfAttention(nn.Module):
    cfg: dict
    w: dict  # q_proj, k_proj, v_proj, o_proj

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        b, t, _ = x.shape
        q = x @ self.w['q_proj'].T
        k = x @ self.w['k_proj'].T
        v = x @ self.w['v_proj'].T
        o_w = self.w['o_proj']

        # heads
        hq = self.cfg['num_attention_heads']
        hk = self.cfg['num_key_value_heads']
        dq = q.shape[-1] // hq
        dk = k.shape[-1] // hk
        q = q.reshape(b, t, hq, dq).transpose(0,2,1,3)
        k = k.reshape(b, t, hk, dk).transpose(0,2,1,3)
        v = v.reshape(b, t, hk, dk).transpose(0,2,1,3)

        # broadcast k/v to all heads
        k = jnp.repeat(k, hq // hk, axis=1)
        v = jnp.repeat(v, hq // hk, axis=1)

        # causal attention
        attn = (q @ k.transpose(0,1,3,2)) / jnp.sqrt(dq)
        mask = jnp.tril(jnp.ones((t, t), dtype=bool))
        attn = nn.softmax(jnp.where(mask, attn, -1e10), axis=-1)

        ctx = (attn @ v).transpose(0,2,1,3).reshape(b, t, o_w.shape[1])
        return ctx @ o_w.T


class BitFFN(nn.Module):
    w: dict  # wi, wo

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hid = x @ self.w['wi'].T
        hid = jnp.square(nn.relu(hid))
        return hid @ self.w['wo'].T


class BitDecoderLayer(nn.Module):
    cfg: dict
    p: dict  # attn, ffn

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        xn = nn.RMSNorm(epsilon=self.cfg['rms_norm_eps'])(x)
        x = x + BitSelfAttention(self.cfg, self.p['attn'])(xn)
        xn = nn.RMSNorm(epsilon=self.cfg['rms_norm_eps'])(x)
        return x + BitFFN(self.p['ffn'])(xn)


class BitNetModel(nn.Module):
    cfg: dict
    params: dict

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
        embed = self.params['embed_tokens']
        x = jnp.take(embed, input_ids, axis=0)

        pos = self.param('pos_embed', nn.initializers.normal(0.02),
                         (1, self.cfg['max_position_embeddings'], self.cfg['hidden_size']))
        x = x + pos[:, :x.shape[1], :]

        for i in range(self.cfg['num_hidden_layers']):
            x = BitDecoderLayer(self.cfg, self.params[f'layer.{i}'])(x)

        x = nn.RMSNorm(epsilon=self.cfg['rms_norm_eps'])(x)
        return x @ embed.T
