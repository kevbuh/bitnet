# bitnet

```bitnet``` is based on Microsoft's [BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), an open-source 1-bit large language model (LLM) with two billion parameters trained on four trillion tokens. 
- **BitLinear**: Drop-in replacement for `nn.Linear` with trainable 1-bit weights.
- **Competitive**: Performs close to 8-bit and FP16 baselines on language tasks.
- **Efficient**: 1-bit weights + activations = low memory + energy use.
- **Scalable**: Follows similar scaling laws to full-precision Transformers.

This repo uses [JAX](https://docs.jax.dev/en/latest/quickstart.html) and [Flax](https://flax.readthedocs.io/en/latest/index.html), and is lightweight enough to work efficiently on a CPU.

<!-- tldr; **No more floats.** Just weights in **[1, 0, -1]**. -->

# Papers

- 04/14/2025 [BitNet Official 2B Parameter Model on Hugging Face](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T) ![NEW](https://img.shields.io/badge/NEW-red)
- 02/18/2025 [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880)
- 11/08/2024 [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965)
- 10/21/2024 [1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs](https://arxiv.org/abs/2410.16144)
- 10/17/2024 bitnet.cpp 1.0 released.
- 03/21/2024 [The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)
- 02/27/2024 [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- 10/17/2023 [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

# Setup

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

# Notes

Notes from [HF model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)

- Parameters: ~2 Billion
- Training Tokens: 4 Trillion
- Context Length: Maximum sequence length of 4096 tokens.
- Transformer-based:
    - modified with BitLinear layers
    - Uses Rotary Position Embeddings (RoPE).
    - Uses squared ReLU (ReLU²) activation in FFN layers.
    - Employs subln normalization.
    - No bias terms in linear or normalization layers.

# Model Architecture

```json
{
    "architectures": [
      "BitNetForCausalLM"
    ],
    "auto_map": {
      "AutoConfig": "configuration_bitnet.BitNetConfig",
      "AutoModelForCausalLM": "modeling_bitnet.BitNetForCausalLM"
    },
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_act": "relu2",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 6912,
    "max_position_embeddings": 4096,
    "model_type": "bitnet",
    "rms_norm_eps": 1e-05,
    "num_attention_heads": 20,
    "num_hidden_layers": 30,
    "num_key_value_heads": 5,
    "rope_theta": 500000.0,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "use_cache": true,
    "vocab_size": 128256,
    "quantization_config": {
      "quant_method": "bitnet",
      "offline_quantization": true
    }
}
```

<!-- # Example

Turns out I really need a GPU to train cuz it takes too long
```bash
(bitnet) ➜  bitnet git:(main) ✗ python bitnet.py
------------
LEARNING RATE: 0.0008
DEVICE: mps
DATA SPLIT: 0.9
MODEL: BitLM
MODEL PARAMS: 3.695681 M
OPTIMIZER: AdamW
------------
loading checkpoint weights
training...
step 34000 | train loss 2.5661 | val loss 2.5680
step 34500 | train loss 2.5614 | val loss 2.5719
step 35000 | train loss 2.5566 | val loss 2.5584
step 35500 | train loss 2.5564 | val loss 2.5602
step 36000 | train loss 2.5588 | val loss 2.5642
step 36500 | train loss 2.5505 | val loss 2.5556
step 37000 | train loss 2.5586 | val loss 2.5518
step 37500 | train loss 2.5515 | val loss 2.5503
step 38000 | train loss 2.5483 | val loss 2.5500
step 38500 | train loss 2.5487 | val loss 2.5553
step 39000 | train loss 2.5498 | val loss 2.5443
step 39500 | train loss 2.5460 | val loss 2.5486
step 40000 | train loss 2.5380 | val loss 2.5454
step 40500 | train loss 2.5401 | val loss 2.5435
step 41000 | train loss 2.5392 | val loss 2.5433
step 41500 | train loss 2.5362 | val loss 2.5416
step 42000 | train loss 2.5355 | val loss 2.5380
step 42500 | train loss 2.5354 | val loss 2.5454
step 43000 | train loss 2.5344 | val loss 2.5415
step 43500 | train loss 2.5336 | val loss 2.5399
WEIGHT VERIFICATION: False
GENERATING TEXT...
GENERATION:  [0, 53, 1, 52, 1, 58, 1, 40, 46, 39, 60, 61, 39, 1, 39, 56, 47, 52, 58, 39, 45, 6, 0, 54, 39, 44, 44, 1, 58, 30, 16, 50, 50, 58, 1, 46, 47, 43, 50, 39, 23, 39, 41, 1, 21, 0, 0, 0, 0, 13, 1, 52, 42, 53, 56, 39, 60, 43, 52, 43, 56, 47, 52, 1, 63, 1, 45, 57, 0, 30, 27, 32, 46, 39, 56, 43, 39, 1, 37, 53, 52, 63, 53, 52, 43, 1, 46, 52, 43, 41, 53, 44, 39, 45, 50, 43, 41, 39, 44, 1, 51]
DECODE:  
tut caetsed lyRAicrst're f, t, mas
Fowuvee d awla, t hay d
ITa beay, poumerem e lder sit owayoud tho
``` -->