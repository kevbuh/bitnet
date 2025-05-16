# bitnet

```bitnet``` is based on Microsoft's [BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), an open-source 1-bit large language model (LLM) with two billion parameters trained on four trillion tokens. 
- **BitLinear**: Drop-in replacement for `nn.Linear` with trainable 1-bit weights.
- **Efficient**: 1-bit weights + activations = low memory + energy use.
- **Scalable**: Follows similar scaling laws to full-precision Transformers.

This repo uses [karpathy's GPT](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py) with the modified nn.Linear layer.

tldr; **No more floats.** Just weights in **[1, 0, -1]**.

# Papers

- 04/14/2025 [BitNet Official 2B Parameter Model on Hugging Face](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
- 02/18/2025 [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880)
- 11/08/2024 [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965)
- 10/21/2024 [1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs](https://arxiv.org/abs/2410.16144)
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

config.json:
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

generation_config.json

```json
{
  "bos_token_id": 128000,
  "eos_token_id": [
    128001,
    128009
  ],
  "do_sample": true,
  "temperature": 0.6,
  "max_length": 4096,
  "top_p": 0.9,
  "transformers_version": "4.40.0.dev0"
}
```

# Todo
- Test performance against huggingface and Microsoft bitnet.cpp
- Make new hardware for it (fpga)
  - https://github.com/rejunity/tiny-asic-1_58bit-matrix-mul
- Make a novel 1-bit Mixture-of-Experts (MoE)
- Set up custom installation script thats nice and says jax or torch and which models to run 
- make {0,1} model work