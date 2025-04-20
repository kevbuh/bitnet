import json
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from safetensors import safe_open
from transformers import GPT2TokenizerFast
from tqdm import tqdm

# --- (1) Load and build model exactly as before ---
from model_utils import load_weight_groups, build_flax_params, BitNetModel

def setup_model(model_dir: str):
    cfg = json.load(open(f"{model_dir}/config.json"))
    groups = load_weight_groups(f"{model_dir}/model.safetensors")
    params = build_flax_params(groups)
    model = BitNetModel(cfg, params)
    return model, cfg, params

model_dir = 'bitnet-b1.58-2B-4T'
model, cfg, params_loaded = setup_model(model_dir)

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
special = json.load(open(f"{model_dir}/special_tokens_map.json"))
tokenizer.add_special_tokens({
    'bos_token': special['bos_token'],
    'eos_token': special['eos_token']
})
tokenizer.pad_token = tokenizer.eos_token
gen_cfg = json.load(open(f"{model_dir}/generation_config.json"))

# Initialize to get flax params (including pos_embed)
key = jax.random.PRNGKey(0)
dummy = jnp.ones((1, 1), dtype=jnp.int32)
vars0 = model.init(key, dummy)
params_only = vars0['params']  # contains 'pos_embed' and all weight arrays

@jax.jit
def step_full(flax_params, input_ids):
    # Forward the full history to use learned pos_embed
    return model.apply({'params': flax_params}, input_ids)

def generate_debug(prompt: str, max_new: int = None) -> str:
    # Tokenize prompt
    ids = tokenizer(prompt, return_tensors='jax')['input_ids']  # shape (1, L)
    eos_ids = set(gen_cfg['eos_token_id'])
    max_new = max_new or gen_cfg['max_length']

    for i in tqdm(range(max_new), desc="Generating"):
        # Get logits for the last position
        logits = step_full(params_only, ids)[0, -1]  # shape (vocab,)

        # Debug: show top-5 tokens + probabilities
        probs = jax.nn.softmax(logits / gen_cfg.get('temperature', 1.0))
        top5 = np.array(jnp.argsort(-probs)[:5])
        top5_probs = np.array(probs[top5])
        top5_tokens = tokenizer.convert_ids_to_tokens(top5.tolist())
        tqdm.write(
            f"Step {i:03d}: " +
            ", ".join(f"{tok}({p:.3f})" for tok, p in zip(top5_tokens, top5_probs))
        )

        # Greedy pick
        nid = int(jnp.argmax(logits))
        if nid in eos_ids:
            tqdm.write(f"⏹ EOS token {nid} at step {i}")
            break

        # Append and continue
        ids = jnp.concatenate([ids, jnp.array([[nid]], dtype=jnp.int32)], axis=1)

    # Decode full sequence
    return tokenizer.decode(np.array(ids[0]), skip_special_tokens=True)

if __name__ == '__main__':
    print("Debug generate:")
    print(generate_debug('<|begin_of_text|>Hello, world'))
