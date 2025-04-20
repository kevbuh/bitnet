import json
import numpy as np
from safetensors import safe_open

import jax
import jax.numpy as jnp
from flax import linen as nn
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from model_utils import load_weight_groups, build_flax_params, BitNetModel


def setup_model(model_dir: str):
    # Load config, weights, and build flax params
    cfg = json.load(open(f"{model_dir}/config.json"))
    groups = load_weight_groups(f"{model_dir}/model.safetensors")
    # Pass cfg to orient/reshape weights correctly
    params = build_flax_params(groups, cfg)
    model = BitNetModel(cfg, params)
    return model, cfg, params

# Instantiate model and tokenizer
d = 'bitnet-b1.58-2B-4T'
model, cfg, params = setup_model(d)

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
special = json.load(open(f"{d}/special_tokens_map.json"))
tokenizer.add_special_tokens({
    'bos_token': special['bos_token'],
    'eos_token': special['eos_token']
})
tokenizer.pad_token = tokenizer.eos_token
gen_cfg = json.load(open(f"{d}/generation_config.json"))

# Initialize model variables and cache
key = jax.random.PRNGKey(0)
dummy = jnp.ones((1,1), dtype=jnp.int32)
vars0 = model.init(key, dummy)
params_only = vars0['params']
cache0 = vars0.get('cache', {})

# JIT-compiled single-step
@jax.jit
def step_one(params, cache, token_ids, key_loc):
    logits, vars_out = model.apply(
        {'params': params, 'cache': cache},
        token_ids,
        mutable=['cache']
    )
    return logits, vars_out['cache'], key_loc

# Generation function

def generate(prompt: str, max_new: int=None) -> str:
    ids0 = tokenizer(prompt, return_tensors='jax')['input_ids']
    batch = ids0.shape[0]
    max_new = max_new or gen_cfg['max_length']

    cache = cache0
    last_token = ids0[:, -1:]
    key_loc = key

    def body(carry, _):
        cache, last_token, key_loc = carry
        lraw, cache, key_loc = step_one(params_only, cache, last_token, key_loc)
        logits = lraw[:, 0, :]
        key_loc, subkey = jax.random.split(key_loc)
        if gen_cfg.get('do_sample', False):
            nid = jax.random.categorical(subkey, logits / gen_cfg.get('temperature', 1.0))
        else:
            nid = jnp.argmax(logits, axis=-1)
        nid = nid.reshape((batch, 1))
        return (cache, nid, key_loc), nid

    (_, _, _), nid_seq = jax.lax.scan(body, (cache, last_token, key_loc), None, length=max_new)
    nid_seq = nid_seq.transpose(1, 0, 2).squeeze(-1)
    final = jnp.concatenate([ids0, nid_seq], axis=1)
    return tokenizer.decode(final[0], skip_special_tokens=True)

# Simple example
if __name__ == '__main__':
    print(generate('<|begin_of_text|>Hello, world'))
