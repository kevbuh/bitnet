# bitnet

```bitnet``` is based on Microsoft's [BitNet b1.58 2B4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T), a binarized LLaMa3-style LLM with 2.4B parameters trained on four trillion tokens. 
<!-- - **BitLinear**: Drop-in replacement for `nn.Linear` with trainable 1-bit weights.
- **Efficient**: 1-bit weights + activations = low memory + energy use.
- **Scalable**: Follows similar scaling laws to full-precision Transformers. -->

tldr; **No more floats.** Just weights in **[1, 0, -1]**.

# Setup

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

# Papers

- 04/14/2025 [BitNet Official 2B Parameter Model on Hugging Face](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
- 02/18/2025 [Bitnet.cpp: Efficient Edge Inference for Ternary LLMs](https://arxiv.org/abs/2502.11880)
- 11/08/2024 [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965)
- 10/21/2024 [1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs](https://arxiv.org/abs/2410.16144)
- 03/21/2024 [The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)
- 02/27/2024 [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- 10/17/2023 [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)

# Notes

Notes from [HF model card](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)

- Parameters: 2,412,820,480 (2.4B)
- Context Length: 4096 tokens
- Weights: 1.58-bit with 8-bit activations (W1.58A8)
- Model: Based off of LLaMa-3
    - modified with BitLinear layers
    - Uses Rotary Position Embeddings [(RoPE)](https://arxiv.org/abs/2104.09864).
    - Uses squared ReLU [(ReLU²)](https://paperswithcode.com/method/squared-relu) activation in FFN layers
    - Employs [Sub-LayerNorm](https://proceedings.mlr.press/v202/wang23u.html) normalization
    - No bias terms in linear or normalization layers
      - Binarization is a form of regularization. By reducing precision, the model generalizes better
- Tokenizer: LLaMA 3 Tokenizer (vocab size: 128,256)
- STE: Straight-through-Estimator to approximate gradients for non-differentiable functions like clip()
- Quantization Function: It first scales the weight matrix by its average absolute value, and then rounds each value to the nearest integer among {-1, 0, +1}
- Binarized LLMs training loss curve follow an S shape

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
    "linear_class": "autobitlinear",
    "quantization_mode": "online"
  }
}
```

Layer Info (2,412,820,480 parameters)

```cs
[Layer name]                                    [Weight shape]             [#Params] [Sample weights]
model.embed_tokens.weight                       torch.Size([128256, 2560]) 328335360 [-0.45703125, 0.90625, 0.69140625, 0.73046875, -0.171875]
model.layers.0.input_layernorm.weight           torch.Size([2560])         2560      [0.0174560546875, 0.0179443359375, 0.019287109375, 0.0274658203125, 0.01300048828125]
model.layers.0.mlp.down_proj.weight             torch.Size([2560, 6912])   17694720  [-1.1328125, -0.46484375, 6.40625, -1.5703125, 0.77734375]
model.layers.0.mlp.ffn_sub_norm.weight          torch.Size([6912])         6912      [1.1875, 1.1953125, 1.3046875, 0.69140625, 3.234375]
model.layers.0.mlp.gate_proj.weight             torch.Size([6912, 2560])   17694720  [0.7734375, 1.84375, 1.15625, -0.6640625, 0.77734375]
model.layers.0.mlp.up_proj.weight               torch.Size([6912, 2560])   17694720  [0.58984375, 2.546875, -1.625, -0.8984375, -5.1875]
model.layers.0.post_attention_layernorm.weight  torch.Size([2560])         2560      [1.34375, 1.3359375, 1.3203125, 1.5703125, 1.2265625]
model.layers.0.self_attn.attn_sub_norm.weight   torch.Size([2560])         2560      [0.0128173828125, 0.0166015625, 0.0152587890625, 0.01513671875, 0.01495361328125]
model.layers.0.self_attn.k_proj.weight          torch.Size([640, 2560])    1638400   [-0.90625, -0.890625, 2.953125, -4.8125, 0.89453125]
model.layers.0.self_attn.o_proj.weight          torch.Size([2560, 2560])   6553600   [-0.458984375, 0.482421875, -4.25, -3.015625, -2.671875]
model.layers.0.self_attn.q_proj.weight          torch.Size([2560, 2560])   6553600   [0.59765625, -0.1904296875, 0.45703125, -2.6875, -0.60546875]
model.layers.0.self_attn.v_proj.weight          torch.Size([640, 2560])    1638400   [7.15625, 1.171875, -0.54296875, 1.1640625, 0.95703125]
model.layers.1.input_layernorm.weight           torch.Size([2560])         2560      [0.016845703125, 0.01531982421875, 0.0172119140625, 0.01409912109375, 0.01611328125]
model.layers.1.mlp.down_proj.weight             torch.Size([2560, 6912])   17694720  [-1.1886598875514971e-34, -2.3773197751029943e-34, 0.63671875, 0.57421875, -4.152786442584977e-34]
model.layers.1.mlp.ffn_sub_norm.weight          torch.Size([6912])         6912      [1.0159280456642669e-32, 1.0785207688568521e-32, 2.28125, 0.4453125, 1.0592614694129797e-32]
model.layers.1.mlp.gate_proj.weight             torch.Size([6912, 2560])   17694720  [-6.229179663877466e-34, -3.2048677980818847e-34, -3.445609041130289e-34, -5.657419211637505e-34, 7.342607912976337e-34]
model.layers.1.mlp.up_proj.weight               torch.Size([6912, 2560])   17694720  [2.3321807920314184e-34, 7.748858760620519e-35, -9.930576275746685e-35, -4.739593222515463e-35, -3.385423730368188e-34]
model.layers.1.post_attention_layernorm.weight  torch.Size([2560])         2560      [1.3203125, 1.328125, 1.203125, 1.234375, 1.1875]
model.layers.1.self_attn.attn_sub_norm.weight   torch.Size([2560])         2560      [0.2060546875, 0.330078125, 0.318359375, 0.2890625, 0.291015625]
model.layers.1.self_attn.k_proj.weight          torch.Size([640, 2560])    1638400   [0.66796875, -4.90625, -0.67578125, -0.0157470703125, 0.6875]
model.layers.1.self_attn.o_proj.weight          torch.Size([2560, 2560])   6553600   [-0.796875, -0.328125, -4.0625, 0.5078125, 3.734375]
model.layers.1.self_attn.q_proj.weight          torch.Size([2560, 2560])   6553600   [-0.1669921875, -0.416015625, -0.1689453125, 0.4140625, 0.40625]
model.layers.1.self_attn.v_proj.weight          torch.Size([640, 2560])    1638400   [1.3046875, -0.006378173828125, 0.076171875, 1.125, 1.125]
model.layers.2.input_layernorm.weight           torch.Size([2560])         2560      [0.0205078125, 0.0184326171875, 0.0166015625, 0.01904296875, 0.0185546875]
model.layers.2.mlp.down_proj.weight             torch.Size([2560, 6912])   17694720  [2.421875, 9.103028252767794e-35, 0.2392578125, -3.325238419606087e-34, 4.78125]
model.layers.2.mlp.ffn_sub_norm.weight          torch.Size([6912])         6912      [0.75390625, 1.1651876163542777e-32, 0.3828125, 1.1700024412152458e-32, 0.8828125]
model.layers.2.mlp.gate_proj.weight             torch.Size([6912, 2560])   17694720  [2.421875, 0.56640625, -0.640625, 0.5546875, -0.255859375]
model.layers.2.mlp.up_proj.weight               torch.Size([6912, 2560])   17694720  [2.1875, 0.51171875, -0.82421875, -0.470703125, 0.50390625]
model.layers.2.post_attention_layernorm.weight  torch.Size([2560])         2560      [1.2890625, 1.296875, 1.140625, 1.2734375, 1.1796875]
model.layers.2.self_attn.attn_sub_norm.weight   torch.Size([2560])         2560      [0.45703125, 0.4921875, 0.45703125, 0.419921875, 0.5]
model.layers.2.self_attn.k_proj.weight          torch.Size([640, 2560])    1638400   [-0.625, -0.189453125, -0.75390625, 2.78125, -2.234375]
model.layers.2.self_attn.o_proj.weight          torch.Size([2560, 2560])   6553600   [2.609375, -4.0, -0.7734375, -0.96484375, 2.25]
model.layers.2.self_attn.q_proj.weight          torch.Size([2560, 2560])   6553600   [0.5, 0.50390625, 0.63671875, 0.423828125, -0.578125]
model.layers.2.self_attn.v_proj.weight          torch.Size([640, 2560])    1638400   [1.0234375, 1.6875, -0.94921875, -0.76953125, -6.5]
model.layers.3.input_layernorm.weight           torch.Size([2560])         2560      [0.021484375, 0.0194091796875, 0.0205078125, 0.0181884765625, 0.018798828125]
model.layers.3.mlp.down_proj.weight             torch.Size([2560, 6912])   17694720  [1.0078125, -3.46875, -0.77734375, 5.34375, 5.4072740137825225e-37]
model.layers.3.mlp.ffn_sub_norm.weight          torch.Size([6912])         6912      [0.65625, 0.890625, 0.921875, 0.921875, 1.1459283169104053e-32]
model.layers.3.mlp.gate_proj.weight             torch.Size([6912, 2560])   17694720  [4.5, -7.9375, 0.875, 4.46875, 0.921875]
model.layers.3.mlp.up_proj.weight               torch.Size([6912, 2560])   17694720  [1.8671875, -0.98046875, -1.6953125, 2.328125, 1.296875]
model.layers.3.post_attention_layernorm.weight  torch.Size([2560])         2560      [1.3046875, 1.359375, 1.1796875, 1.3125, 1.21875]
model.layers.3.self_attn.attn_sub_norm.weight   torch.Size([2560])         2560      [0.4140625, 0.31640625, 0.39453125, 0.38671875, 0.419921875]
model.layers.3.self_attn.k_proj.weight          torch.Size([640, 2560])    1638400   [0.59765625, 0.002410888671875, 0.1875, 0.765625, 0.546875]
model.layers.3.self_attn.o_proj.weight          torch.Size([2560, 2560])   6553600   [-1.265625, 0.765625, -0.9765625, 3.34375, -5.5]
model.layers.3.self_attn.q_proj.weight          torch.Size([2560, 2560])   6553600   [0.7265625, 0.515625, -5.5, -0.4765625, 0.486328125]
model.layers.3.self_attn.v_proj.weight          torch.Size([640, 2560])    1638400   [1.390625, 4.8125, -1.25, 1.3515625, -5.34375]
model.layers.4.input_layernorm.weight           torch.Size([2560])         2560      [0.0186767578125, 0.0185546875, 0.0177001953125, 0.019775390625, 0.0162353515625]
model.layers.4.mlp.down_proj.weight             torch.Size([2560, 6912])   17694720  [1.0703125, -1.078125, 2.90625, -0.84765625, -0.9453125]
model.layers.4.mlp.ffn_sub_norm.weight          torch.Size([6912])         6912      [0.7421875, 0.2314453125, 0.5390625, 0.8984375, 1.0390625]
model.layers.4.mlp.gate_proj.weight             torch.Size([6912, 2560])   17694720  [-0.1650390625, 1.046875, -2.90625, -1.0546875, -0.353515625]
model.layers.4.mlp.up_proj.weight               torch.Size([6912, 2560])   17694720  [2.484375, 0.75, -0.9765625, -0.294921875, -4.25]
model.layers.4.post_attention_layernorm.weight  torch.Size([2560])         2560      [1.3125, 1.3671875, 1.2109375, 1.3046875, 1.2109375]
model.layers.4.self_attn.attn_sub_norm.weight   torch.Size([2560])         2560      [0.322265625, 0.302734375, 0.357421875, 0.3984375, 0.26953125]
model.layers.4.self_attn.k_proj.weight          torch.Size([640, 2560])    1638400   [0.90625, 1.0390625, 0.7421875, 0.5703125, -1.6953125]
model.layers.4.self_attn.o_proj.weight          torch.Size([2560, 2560])   6553600   [-2.5625, 1.4140625, 1.0625, -1.0703125, -1.265625]
model.layers.4.self_attn.q_proj.weight          torch.Size([2560, 2560])   6553600   [-0.5, -0.1416015625, -0.01458740234375, 0.46484375, 0.47265625]
model.layers.4.self_attn.v_proj.weight          torch.Size([640, 2560])    1638400   [1.515625, -1.53125, -2.0, 1.6171875, -1.8046875]
model.layers.5.input_layernorm.weight           torch.Size([2560])         2560      [0.0155029296875, 0.015869140625, 0.01611328125, 0.0145263671875, 0.01507568359375]
model.layers.5.mlp.down_proj.weight             torch.Size([2560, 6912])   17694720  [0.057373046875, -7.28125, 1.921875, 3.765625, -0.8125]
model.layers.5.mlp.ffn_sub_norm.weight          torch.Size([6912])         6912      [1.84375, 0.94921875, 0.70703125, 1.046875, 1.078125]
model.layers.5.mlp.gate_proj.weight             torch.Size([6912, 2560])   17694720  [0.90625, 1.6171875, 3.546875, -3.640625, 1.140625]
model.layers.5.mlp.up_proj.weight               torch.Size([6912, 2560])   17694720  [2.96875, 1.0078125, -0.11767578125, -0.67578125, 3.875]
model.layers.5.post_attention_layernorm.weight  torch.Size([2560])         2560      [1.34375, 1.3984375, 1.2265625, 1.34375, 1.2421875]
model.layers.5.self_attn.attn_sub_norm.weight   torch.Size([2560])         2560      [0.625, 0.5546875, 0.546875, 0.64453125, 0.5546875]
model.layers.5.self_attn.k_proj.weight          torch.Size([640, 2560])    1638400   [-0.97265625, -6.75, -0.80859375, -0.88671875, 0.97265625]
model.layers.5.self_attn.o_proj.weight          torch.Size([2560, 2560])   6553600   [-2.515625, -1.046875, -4.34375, -1.0859375, 1.0625]
model.layers.5.self_attn.q_proj.weight          torch.Size([2560, 2560])   6553600   [0.72265625, 0.6328125, -0.4609375, -0.54296875, -0.6484375]
model.layers.5.self_attn.v_proj.weight          torch.Size([640, 2560])    1638400   [0.1767578125, 1.3046875, -7.375, 4.46875, -4.28125]
model.layers.6.input_layernorm.weight           torch.Size([2560])         2560      [0.017822265625, 0.0159912109375, 0.0184326171875, 0.0179443359375, 0.016357421875]
model.layers.6.mlp.down_proj.weight             torch.Size([2560, 6912])   17694720  [-1.1640625, 0.025634765625, 1.140625, -3.015625, 0.8359375]
model.layers.6.mlp.ffn_sub_norm.weight          torch.Size([6912])         6912      [0.5625, 0.271484375, 1.640625, 0.1826171875, 0.53125]
model.layers.6.mlp.gate_proj.weight             torch.Size([6912, 2560])   17694720  [-4.28125, 1.0390625, -0.765625, 1.3984375, -6.78125]
model.layers.6.mlp.up_proj.weight               torch.Size([6912, 2560])   17694720  [-0.4296875, -0.91015625, -0.3046875, 0.5859375, 0.267578125]
model.layers.6.post_attention_layernorm.weight  torch.Size([2560])         2560      [1.3515625, 1.421875, 1.296875, 1.359375, 1.2421875]
model.layers.6.self_attn.attn_sub_norm.weight   torch.Size([2560])         2560      [0.64453125, 0.6484375, 0.58203125, 0.64453125, 0.60546875]
model.layers.6.self_attn.k_proj.weight          torch.Size([640, 2560])    1638400   [-1.09375, 0.478515625, -1.0625, 0.283203125, -1.078125]
model.layers.6.self_attn.o_proj.weight          torch.Size([2560, 2560])   6553600   [-1.3515625, 0.51171875, 1.171875, 0.65625, 1.1796875]
model.layers.6.self_attn.q_proj.weight          torch.Size([2560, 2560])   6553600   [0.5546875, 2.0625, 0.67578125, 0.80859375, 0.671875]
model.layers.6.self_attn.v_proj.weight          torch.Size([640, 2560])    1638400   [1.734375, 1.234375, -1.71875, -0.470703125, -1.7421875]
model.layers.7.input_layernorm.weight           torch.Size([2560])         2560      [0.0166015625, 0.0157470703125, 0.0150146484375, 0.015869140625, 0.01495361328125]
model.layers.7.mlp.down_proj.weight             torch.Size([2560, 6912])   17694720  [1.1953125, 1.1875, -3.109375, 0.2421875, -0.138671875]
model.layers.7.mlp.ffn_sub_norm.weight          torch.Size([6912])         6912      [0.734375, 1.6015625, 1.4609375, 0.98046875, 1.0390625]
model.layers.7.mlp.gate_proj.weight             torch.Size([6912, 2560])   17694720  [-2.046875, 0.98046875, 1.015625, 0.9609375, 0.11669921875]
model.layers.7.mlp.up_proj.weight               torch.Size([6912, 2560])   17694720  [0.765625, 0.875, 1.0703125, -1.296875, -2.5]
model.layers.7.post_attention_layernorm.weight  torch.Size([2560])         2560      [1.359375, 1.3984375, 1.3203125, 1.359375, 1.25]
model.layers.7.self_attn.attn_sub_norm.weight   torch.Size([2560])         2560      [0.7109375, 0.77734375, 0.8359375, 0.80078125, 0.828125]
model.layers.7.self_attn.k_proj.weight          torch.Size([640, 2560])    1638400   [-0.412109375, -1.125, -1.140625, -0.86328125, 0.5546875]
model.layers.7.self_attn.o_proj.weight          torch.Size([2560, 2560])   6553600   [-3.796875, -0.85546875, -12.4375, -1.125, 2.953125]
model.layers.7.self_attn.q_proj.weight          torch.Size([2560, 2560])   6553600   [-0.80859375, 0.396484375, -0.703125, -0.671875, -0.265625]
model.layers.7.self_attn.v_proj.weight          torch.Size([640, 2560])    1638400   [-2.921875, -0.9609375, 1.6171875, 0.59375, -1.6015625]
model.layers.8.input_layernorm.weight           torch.Size([2560])         2560      [0.0169677734375, 0.01708984375, 0.0166015625, 0.0167236328125, 0.01513671875]
model.layers.8.mlp.down_proj.weight             torch.Size([2560, 6912])   17694720  [6.8125, 1.2734375, -1.171875, 5.0, -1.3125]
model.layers.8.mlp.ffn_sub_norm.weight          torch.Size([6912])         6912      [1.0859375, 0.51171875, 0.90234375, 0.5078125, 0.95703125]
model.layers.8.mlp.gate_proj.weight             torch.Size([6912, 2560])   17694720  [-2.140625, 1.1328125, -0.65625, -0.1025390625, 0.6875]
model.layers.8.mlp.up_proj.weight               torch.Size([6912, 2560])   17694720  [0.9453125, -3.890625, 0.84765625, -0.94921875, -3.1875]
model.layers.8.post_attention_layernorm.weight  torch.Size([2560])         2560      [1.359375, 1.375, 1.3046875, 1.359375, 1.234375]
model.layers.8.self_attn.attn_sub_norm.weight   torch.Size([2560])         2560      [0.90625, 0.84375, 0.9296875, 0.87890625, 0.89453125]
model.layers.8.self_attn.k_proj.weight          torch.Size([640, 2560])    1638400   [-1.140625, 1.0703125, -0.11865234375, 1.7265625, 1.140625]
model.layers.8.self_attn.o_proj.weight          torch.Size([2560, 2560])   6553600   [-1.09375, -1.3203125, 0.439453125, -1.3125, -3.703125]
model.layers.8.self_attn.q_proj.weight          torch.Size([2560, 2560])   6553600   [0.50390625, 0.78515625, 0.671875, 0.57421875, 0.7265625]
model.layers.8.self_attn.v_proj.weight          torch.Size([640, 2560])    1638400   [2.5, 7.75, 13.125, 7.3125, -8.375]
model.layers.9.input_layernorm.weight           torch.Size([2560])         2560      [0.01300048828125, 0.01470947265625, 0.01263427734375, 0.0152587890625, 0.0123291015625]
model.layers.9.mlp.down_proj.weight             torch.Size([2560, 6912])   17694720  [1.203125, -3.46875, -1.3125, -1.6796875, -1.3125]
model.layers.9.mlp.ffn_sub_norm.weight          torch.Size([6912])         6912      [1.046875, 3.8125, 2.546875, 0.83984375, 1.9609375]
model.layers.9.mlp.gate_proj.weight             torch.Size([6912, 2560])   17694720  [-0.69921875, 1.09375, 8.0, 0.92578125, -2.0]
model.layers.9.mlp.up_proj.weight               torch.Size([6912, 2560])   17694720  [6.8125, 0.95703125, -1.6328125, 2.25, 1.078125]
model.layers.9.post_attention_layernorm.weight  torch.Size([2560])         2560      [1.296875, 1.2578125, 1.28125, 1.3203125, 1.1875]
model.layers.9.self_attn.attn_sub_norm.weight   torch.Size([2560])         2560      [0.8125, 0.83203125, 0.94140625, 0.84375, 0.8125]
model.layers.9.self_attn.k_proj.weight          torch.Size([640, 2560])    1638400   [-0.1396484375, -1.0234375, -1.1640625, -1.171875, 1.1015625]
model.layers.9.self_attn.o_proj.weight          torch.Size([2560, 2560])   6553600   [0.96484375, 2.375, -6.375, -0.93359375, 10.25]
model.layers.9.self_attn.q_proj.weight          torch.Size([2560, 2560])   6553600   [-0.7421875, 4.46875, 0.66015625, 2.53125, -0.5625]
model.layers.9.self_attn.v_proj.weight          torch.Size([640, 2560])    1638400   [10.125, 1.8125, 1.8125, 6.90625, 9.25]
model.layers.10.input_layernorm.weight          torch.Size([2560])         2560      [0.0172119140625, 0.01556396484375, 0.013916015625, 0.015869140625, 0.013427734375]
model.layers.10.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [-3.9375, 10.875, -0.31640625, -0.89453125, -1.1328125]
model.layers.10.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [3.265625, 1.3828125, 0.7578125, 1.3515625, 1.171875]
model.layers.10.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-0.73828125, -0.78515625, -0.283203125, -6.09375, 1.3125]
model.layers.10.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [-5.34375, 0.7421875, 0.91015625, -2.25, 0.98046875]
model.layers.10.post_attention_layernorm.weight torch.Size([2560])         2560      [1.296875, 1.265625, 1.28125, 1.328125, 1.21875]
model.layers.10.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [0.91015625, 0.92578125, 0.921875, 0.890625, 0.875]
model.layers.10.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [0.625, -0.609375, 1.25, 0.0791015625, 1.265625]
model.layers.10.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [-0.369140625, 1.3125, -6.78125, -1.28125, 7.8125]
model.layers.10.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [0.396484375, -0.76953125, 0.1005859375, 0.35546875, 0.78125]
model.layers.10.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [1.90625, -1.9140625, -1.9140625, 1.921875, -3.84375]
model.layers.11.input_layernorm.weight          torch.Size([2560])         2560      [0.01397705078125, 0.0157470703125, 0.0152587890625, 0.0172119140625, 0.0130615234375]
model.layers.11.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [-0.8515625, -5.875, 1.2421875, 1.234375, -1.1484375]
model.layers.11.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [1.5234375, 0.94140625, 1.71875, 0.66015625, 1.609375]
model.layers.11.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-1.03125, -3.796875, -1.0078125, -4.0, -1.3359375]
model.layers.11.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [3.421875, 6.40625, -1.015625, -1.1875, -1.0390625]
model.layers.11.post_attention_layernorm.weight torch.Size([2560])         2560      [1.3125, 1.3359375, 1.3125, 1.390625, 1.2421875]
model.layers.11.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [2.03125, 2.46875, 2.171875, 2.3125, 2.265625]
model.layers.11.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [-1.1640625, -1.265625, -1.2265625, -1.265625, 1.296875]
model.layers.11.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [6.5, 1.5625, -1.359375, -1.375, -1.5078125]
model.layers.11.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [0.5234375, 0.259765625, 0.75390625, -0.6796875, -0.61328125]
model.layers.11.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [2.53125, -0.0927734375, 0.482421875, -3.890625, -1.9921875]
model.layers.12.input_layernorm.weight          torch.Size([2560])         2560      [0.01373291015625, 0.01373291015625, 0.01422119140625, 0.0137939453125, 0.01220703125]
model.layers.12.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [0.875, -0.5, 1.296875, 9.375, -2.46875]
model.layers.12.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [2.265625, 1.6796875, 1.34375, 1.8359375, 0.74609375]
model.layers.12.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-0.94140625, -2.1875, 2.34375, -1.0390625, 3.46875]
model.layers.12.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [0.75, -0.96875, 1.28125, -0.80078125, -1.015625]
model.layers.12.post_attention_layernorm.weight torch.Size([2560])         2560      [1.3125, 1.34375, 1.28125, 1.40625, 1.203125]
model.layers.12.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [1.2734375, 1.375, 1.3984375, 1.3125, 1.3515625]
model.layers.12.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [1.0546875, -0.84765625, 0.408203125, -1.3828125, -1.1953125]
model.layers.12.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [-1.765625, 7.0, 0.87109375, 1.5703125, 8.75]
model.layers.12.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [-0.66015625, -0.828125, -0.6328125, 0.95703125, -0.91015625]
model.layers.12.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-0.1083984375, 0.51171875, -1.9453125, -2.734375, -2.21875]
model.layers.13.input_layernorm.weight          torch.Size([2560])         2560      [0.01336669921875, 0.0133056640625, 0.01318359375, 0.0133056640625, 0.01214599609375]
model.layers.13.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [-5.9375, 0.98046875, -1.453125, 4.375, -1.21875]
model.layers.13.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [2.671875, 2.21875, 2.390625, 1.203125, 2.734375]
model.layers.13.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-1.421875, -1.046875, -1.1328125, 3.515625, -3.03125]
model.layers.13.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [0.734375, -2.921875, 0.96875, -1.3515625, 1.03125]
model.layers.13.post_attention_layernorm.weight torch.Size([2560])         2560      [1.234375, 1.2421875, 1.28125, 1.3515625, 1.171875]
model.layers.13.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [1.53125, 1.484375, 1.515625, 1.3828125, 1.5234375]
model.layers.13.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [0.059326171875, 1.265625, -1.25, 1.2421875, -0.39453125]
model.layers.13.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [0.462890625, 1.6875, 16.25, -1.75, -4.4375]
model.layers.13.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [-0.298828125, 0.8125, 0.49609375, 0.76953125, -0.8359375]
model.layers.13.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-0.578125, 2.078125, -1.9296875, 6.09375, 2.09375]
model.layers.14.input_layernorm.weight          torch.Size([2560])         2560      [0.01336669921875, 0.0135498046875, 0.01422119140625, 0.01458740234375, 0.01324462890625]
model.layers.14.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [1.1328125, 1.25, 1.09375, 10.75, 0.32421875]
model.layers.14.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [2.890625, 2.125, 1.6015625, 2.8125, 2.390625]
model.layers.14.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-0.1875, 1.2734375, 0.71484375, 0.96875, -1.140625]
model.layers.14.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [3.625, 1.203125, 3.34375, -0.76171875, -1.515625]
model.layers.14.post_attention_layernorm.weight torch.Size([2560])         2560      [1.2578125, 1.265625, 1.2578125, 1.3828125, 1.1484375]
model.layers.14.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [2.453125, 2.21875, 2.171875, 2.25, 2.375]
model.layers.14.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [-0.1650390625, 1.5, -1.203125, 0.30078125, 1.4140625]
model.layers.14.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [1.296875, 1.25, 8.9375, -4.875, -5.25]
model.layers.14.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [0.9140625, -0.09228515625, -0.6015625, -0.42578125, 0.400390625]
model.layers.14.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-2.09375, -3.875, -7.25, 4.28125, -18.0]
model.layers.15.input_layernorm.weight          torch.Size([2560])         2560      [0.01214599609375, 0.0157470703125, 0.01214599609375, 0.012939453125, 0.01153564453125]
model.layers.15.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [-3.28125, -1.3046875, -1.4921875, 2.15625, 4.34375]
model.layers.15.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [3.46875, 1.6015625, 1.4921875, 3.140625, 1.4609375]
model.layers.15.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-1.078125, -1.078125, -7.21875, 9.1875, -0.31640625]
model.layers.15.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [-1.046875, 1.0703125, 7.4375, 1.03125, 0.62109375]
model.layers.15.post_attention_layernorm.weight torch.Size([2560])         2560      [1.359375, 1.421875, 1.3828125, 1.484375, 1.296875]
model.layers.15.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [1.765625, 1.9375, 1.609375, 2.0625, 2.046875]
model.layers.15.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [1.2109375, -0.7578125, -1.359375, 1.3671875, -1.171875]
model.layers.15.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [1.421875, 3.640625, 3.625, -1.4140625, -1.3984375]
model.layers.15.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [-0.83203125, 0.1923828125, -0.83984375, -0.5390625, -0.84765625]
model.layers.15.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-5.65625, -0.79296875, 8.375, -2.25, -2.25]
model.layers.16.input_layernorm.weight          torch.Size([2560])         2560      [0.0120849609375, 0.01190185546875, 0.01080322265625, 0.0128173828125, 0.010009765625]
model.layers.16.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [-1.3046875, 12.0, 1.3203125, -3.5625, 5.34375]
model.layers.16.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [2.34375, 3.109375, 1.9921875, 1.90625, 4.8125]
model.layers.16.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [5.65625, -1.109375, 0.62109375, -0.80859375, -5.3125]
model.layers.16.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [-6.90625, -1.03125, 7.1875, -0.90234375, 0.7890625]
model.layers.16.post_attention_layernorm.weight torch.Size([2560])         2560      [1.3671875, 1.4453125, 1.3671875, 1.453125, 1.296875]
model.layers.16.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [1.921875, 1.9296875, 1.9453125, 1.9453125, 2.0]
model.layers.16.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [-1.5390625, -1.2578125, 1.5625, 1.515625, -0.4765625]
model.layers.16.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [1.390625, 5.3125, -1.40625, -3.296875, -1.21875]
model.layers.16.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [-2.40625, -1.0078125, -0.921875, -0.455078125, -1.0234375]
model.layers.16.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [3.484375, -2.140625, 3.328125, 10.4375, -4.5]
model.layers.17.input_layernorm.weight          torch.Size([2560])         2560      [0.01214599609375, 0.0126953125, 0.01275634765625, 0.0125732421875, 0.01263427734375]
model.layers.17.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [-0.8359375, -12.875, -1.9296875, 6.34375, 1.34375]
model.layers.17.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [3.234375, 3.140625, 2.671875, 1.8515625, 2.171875]
model.layers.17.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-1.21875, 0.50390625, 0.8671875, -1.109375, 1.203125]
model.layers.17.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [2.484375, -1.6875, -1.0546875, -0.69140625, -3.578125]
model.layers.17.post_attention_layernorm.weight torch.Size([2560])         2560      [1.3828125, 1.4765625, 1.3984375, 1.484375, 1.3046875]
model.layers.17.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [2.625, 2.625, 2.46875, 2.59375, 2.65625]
model.layers.17.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [1.40625, -7.96875, -1.703125, -1.421875, 1.2109375]
model.layers.17.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [-2.234375, 1.609375, 4.3125, -3.484375, -1.5]
model.layers.17.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [0.6640625, -0.9765625, 0.76953125, 0.890625, 0.9765625]
model.layers.17.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-6.375, 2.140625, 2.296875, -14.125, 3.375]
model.layers.18.input_layernorm.weight          torch.Size([2560])         2560      [0.01336669921875, 0.01531982421875, 0.01226806640625, 0.0147705078125, 0.0135498046875]
model.layers.18.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [0.4453125, -12.75, -3.640625, 1.578125, 3.640625]
model.layers.18.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [1.7578125, 4.84375, 4.40625, 3.890625, 3.71875]
model.layers.18.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-1.3515625, -2.296875, -1.296875, 1.546875, 1.1484375]
model.layers.18.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [1.375, 0.9375, 2.328125, 0.89453125, 1.09375]
model.layers.18.post_attention_layernorm.weight torch.Size([2560])         2560      [1.4140625, 1.5234375, 1.4609375, 1.546875, 1.375]
model.layers.18.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [5.03125, 4.96875, 5.15625, 5.28125, 4.0625]
model.layers.18.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [-0.031982421875, -3.390625, 1.109375, -1.2578125, -1.28125]
model.layers.18.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [1.5234375, -5.75, 1.0234375, -1.3203125, 1.3125]
model.layers.18.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [0.328125, -2.296875, -0.291015625, -0.0400390625, -0.71484375]
model.layers.18.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-7.46875, 0.52734375, 2.140625, -3.21875, 1.484375]
model.layers.19.input_layernorm.weight          torch.Size([2560])         2560      [0.013916015625, 0.01263427734375, 0.0146484375, 0.015380859375, 0.01409912109375]
model.layers.19.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [0.21875, -1.1328125, -9.75, -8.625, 3.671875]
model.layers.19.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [2.90625, 3.109375, 4.65625, 6.25, 5.375]
model.layers.19.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-2.609375, 1.296875, 1.1875, 7.0, -10.0]
model.layers.19.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [0.98046875, -1.03125, -6.875, 1.2578125, -7.8125]
model.layers.19.post_attention_layernorm.weight torch.Size([2560])         2560      [1.421875, 1.53125, 1.4296875, 1.5390625, 1.3515625]
model.layers.19.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [3.234375, 3.296875, 3.125, 3.34375, 3.3125]
model.layers.19.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [1.1875, -2.0625, -1.265625, -1.15625, 1.1953125]
model.layers.19.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [-3.078125, 3.75, -1.375, 0.5390625, -4.84375]
model.layers.19.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [-0.8046875, -0.9609375, -0.82421875, -0.462890625, 0.8125]
model.layers.19.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-2.21875, -0.078125, 6.53125, -9.5625, 14.5]
model.layers.20.input_layernorm.weight          torch.Size([2560])         2560      [0.01287841796875, 0.01409912109375, 0.013916015625, 0.01422119140625, 0.01226806640625]
model.layers.20.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [-1.2109375, -1.09375, -1.71875, -0.93359375, 1.25]
model.layers.20.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [3.9375, 5.4375, 6.75, 1.9375, 4.96875]
model.layers.20.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-0.94140625, 1.1640625, 1.15625, -1.15625, -1.265625]
model.layers.20.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [-1.3359375, 2.65625, -0.65625, 1.59375, -2.0625]
model.layers.20.post_attention_layernorm.weight torch.Size([2560])         2560      [1.4140625, 1.484375, 1.4609375, 1.546875, 1.3671875]
model.layers.20.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [4.84375, 4.53125, 4.625, 4.34375, 4.34375]
model.layers.20.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [-1.6640625, -1.4609375, 0.63671875, -1.4921875, 1.609375]
model.layers.20.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [-1.0078125, -4.09375, 2.734375, 6.6875, -1.234375]
model.layers.20.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [-0.9140625, -0.62890625, -0.91796875, -0.8359375, -0.97265625]
model.layers.20.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [9.6875, -4.09375, 2.109375, -3.640625, -1.9765625]
model.layers.21.input_layernorm.weight          torch.Size([2560])         2560      [0.012939453125, 0.01312255859375, 0.01312255859375, 0.0137939453125, 0.01312255859375]
model.layers.21.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [7.03125, -13.4375, -1.4140625, -2.21875, -3.234375]
model.layers.21.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [3.78125, 8.625, 3.703125, 5.21875, 6.96875]
model.layers.21.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-0.85546875, -2.375, -0.296875, 4.65625, -1.203125]
model.layers.21.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [1.7265625, -1.9140625, 7.4375, -1.46875, -0.7890625]
model.layers.21.post_attention_layernorm.weight torch.Size([2560])         2560      [1.421875, 1.484375, 1.4453125, 1.5234375, 1.359375]
model.layers.21.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [5.53125, 5.28125, 5.5, 5.65625, 5.5]
model.layers.21.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [-1.5390625, -8.4375, 0.46875, -1.390625, -1.1796875]
model.layers.21.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [1.140625, 1.4375, 1.296875, 1.234375, 1.1484375]
model.layers.21.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [0.447265625, 0.82421875, -0.42578125, 1.09375, 0.062255859375]
model.layers.21.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [5.96875, 1.9140625, -1.203125, -1.90625, -1.9140625]
model.layers.22.input_layernorm.weight          torch.Size([2560])         2560      [0.01483154296875, 0.01373291015625, 0.01513671875, 0.01458740234375, 0.01556396484375]
model.layers.22.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [2.0, 6.34375, 4.09375, -5.46875, 1.4375]
model.layers.22.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [4.59375, 4.625, 10.4375, 3.03125, 4.875]
model.layers.22.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [5.3125, 3.6875, 2.515625, -2.796875, 1.203125]
model.layers.22.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [-6.125, -4.875, -1.5859375, 1.5, 1.1328125]
model.layers.22.post_attention_layernorm.weight torch.Size([2560])         2560      [1.4296875, 1.515625, 1.4375, 1.5546875, 1.40625]
model.layers.22.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [4.65625, 4.25, 4.46875, 2.65625, 4.15625]
model.layers.22.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [1.765625, 0.9140625, -0.1728515625, 1.1875, -2.03125]
model.layers.22.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [1.2265625, -3.921875, -1.2578125, -1.8515625, -1.28125]
model.layers.22.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [-0.546875, -0.5390625, -3.375, 0.75390625, -0.03955078125]
model.layers.22.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-0.416015625, -1.1875, 10.3125, 1.890625, -4.5625]
model.layers.23.input_layernorm.weight          torch.Size([2560])         2560      [0.01324462890625, 0.01300048828125, 0.0128173828125, 0.01416015625, 0.01470947265625]
model.layers.23.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [-1.0, 1.4609375, 0.003875732421875, -0.77734375, -13.4375]
model.layers.23.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [4.59375, 5.78125, 7.71875, 8.625, 10.5625]
model.layers.23.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [6.875, -1.1953125, -1.203125, -1.5703125, -1.4140625]
model.layers.23.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [-12.5, 2.09375, -1.125, 4.125, 0.7578125]
model.layers.23.post_attention_layernorm.weight torch.Size([2560])         2560      [1.484375, 1.5390625, 1.4609375, 1.5859375, 1.4375]
model.layers.23.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [6.5625, 6.09375, 6.3125, 6.28125, 6.65625]
model.layers.23.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [-1.3125, 1.3359375, -1.3984375, -1.3046875, 0.81640625]
model.layers.23.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [5.3125, -1.359375, 11.0625, -0.9375, 1.40625]
model.layers.23.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [0.8359375, 0.44140625, 0.48046875, -2.421875, -2.15625]
model.layers.23.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-8.75, 1.828125, -7.15625, 1.953125, -1.8515625]
model.layers.24.input_layernorm.weight          torch.Size([2560])         2560      [0.0130615234375, 0.01190185546875, 0.01422119140625, 0.013671875, 0.01470947265625]
model.layers.24.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [1.71875, -1.453125, 5.25, -1.4609375, 10.875]
model.layers.24.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [5.0625, 5.59375, 7.3125, 8.0625, 8.3125]
model.layers.24.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [5.15625, 5.0, -3.265625, 1.1484375, 1.890625]
model.layers.24.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [1.09375, 1.109375, -1.4296875, 0.049072265625, 1.8828125]
model.layers.24.post_attention_layernorm.weight torch.Size([2560])         2560      [1.4921875, 1.5078125, 1.4921875, 1.5390625, 1.4453125]
model.layers.24.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [8.0, 8.4375, 7.8125, 7.90625, 7.34375]
model.layers.24.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [0.71875, -0.85546875, 1.6640625, -1.5625, -0.2412109375]
model.layers.24.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [-1.3984375, 1.390625, 1.3828125, -6.40625, 9.0625]
model.layers.24.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [0.023193359375, -0.80859375, -0.302734375, -0.67578125, -0.953125]
model.layers.24.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-3.90625, -7.78125, -13.125, 9.0625, 1.859375]
model.layers.25.input_layernorm.weight          torch.Size([2560])         2560      [0.0172119140625, 0.01513671875, 0.0157470703125, 0.01953125, 0.017333984375]
model.layers.25.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [1.2890625, 0.72265625, 0.443359375, -11.3125, 1.46875]
model.layers.25.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [7.34375, 4.03125, 3.921875, 5.90625, 7.5625]
model.layers.25.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-1.3125, 0.703125, 1.703125, -2.34375, -1.3828125]
model.layers.25.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [-3.453125, 0.8984375, -4.375, -4.84375, -9.8125]
model.layers.25.post_attention_layernorm.weight torch.Size([2560])         2560      [1.4609375, 1.5078125, 1.4296875, 1.53125, 1.390625]
model.layers.25.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [6.6875, 5.71875, 7.28125, 7.21875, 8.5625]
model.layers.25.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [-1.15625, -1.171875, -3.75, 1.328125, 1.1796875]
model.layers.25.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [-1.0390625, 1.4140625, 1.359375, -2.40625, 1.0390625]
model.layers.25.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [-1.1015625, -1.59375, 0.75390625, 0.64453125, -0.12890625]
model.layers.25.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-1.671875, -1.6875, -4.15625, -3.09375, -1.6796875]
model.layers.26.input_layernorm.weight          torch.Size([2560])         2560      [0.0150146484375, 0.013916015625, 0.01544189453125, 0.015625, 0.01556396484375]
model.layers.26.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [-3.65625, 1.3671875, 0.76953125, -2.234375, 1.2265625]
model.layers.26.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [5.46875, 11.3125, 9.125, 6.78125, 7.0]
model.layers.26.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [1.109375, -0.55078125, 3.875, -1.203125, 4.125]
model.layers.26.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [2.453125, -4.65625, 0.185546875, 1.1875, 0.056396484375]
model.layers.26.post_attention_layernorm.weight torch.Size([2560])         2560      [1.453125, 1.4453125, 1.453125, 1.546875, 1.453125]
model.layers.26.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [9.6875, 9.0, 9.0, 9.125, 9.5]
model.layers.26.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [0.5234375, -1.265625, -1.0859375, 1.390625, -1.21875]
model.layers.26.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [0.88671875, 8.375, -1.421875, 3.5625, -4.875]
model.layers.26.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [0.37890625, -0.8203125, -0.7890625, 0.66015625, 1.21875]
model.layers.26.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [9.625, 1.625, 17.875, 1.7421875, -1.4921875]
model.layers.27.input_layernorm.weight          torch.Size([2560])         2560      [0.015869140625, 0.01556396484375, 0.0169677734375, 0.017578125, 0.0167236328125]
model.layers.27.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [13.625, -0.0115966796875, 0.349609375, -1.40625, -1.2109375]
model.layers.27.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [8.875, 7.375, 8.375, 2.765625, 3.78125]
model.layers.27.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-1.2578125, 1.265625, -0.78125, -1.234375, 1.640625]
model.layers.27.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [2.109375, 3.375, 1.09375, 3.25, -6.09375]
model.layers.27.post_attention_layernorm.weight torch.Size([2560])         2560      [1.5390625, 1.5390625, 1.5234375, 1.609375, 1.5078125]
model.layers.27.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [9.9375, 9.8125, 10.375, 10.1875, 10.125]
model.layers.27.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [-1.2734375, -1.296875, -1.2890625, 3.71875, -0.9921875]
model.layers.27.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [-0.74609375, 5.46875, 1.328125, -3.65625, -0.90234375]
model.layers.27.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [-0.7890625, 0.203125, 0.205078125, 0.55078125, 0.76953125]
model.layers.27.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [1.8515625, 1.859375, -1.953125, 4.25, 1.28125]
model.layers.28.input_layernorm.weight          torch.Size([2560])         2560      [0.021240234375, 0.01556396484375, 0.0181884765625, 0.0206298828125, 0.0194091796875]
model.layers.28.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [2.015625, -1.4140625, 5.84375, 1.2890625, -0.455078125]
model.layers.28.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [4.0, 12.25, 12.0625, 10.4375, 4.4375]
model.layers.28.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-1.3359375, -9.8125, -0.94921875, 1.6015625, -0.88671875]
model.layers.28.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [-5.375, -10.8125, -4.15625, 5.4375, -1.9140625]
model.layers.28.post_attention_layernorm.weight torch.Size([2560])         2560      [1.5390625, 1.515625, 1.484375, 1.5234375, 1.484375]
model.layers.28.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [11.6875, 7.625, 13.0, 11.375, 11.4375]
model.layers.28.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [0.3359375, 1.03125, -0.57421875, -0.765625, 1.265625]
model.layers.28.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [2.140625, 3.1875, 0.9296875, -0.92578125, 0.6953125]
model.layers.28.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [-0.470703125, 0.6171875, 0.609375, 2.546875, -0.376953125]
model.layers.28.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [4.5, 1.5078125, -4.21875, 5.21875, -2.8125]
model.layers.29.input_layernorm.weight          torch.Size([2560])         2560      [0.0233154296875, 0.02490234375, 0.0216064453125, 0.0186767578125, 0.021728515625]
model.layers.29.mlp.down_proj.weight            torch.Size([2560, 6912])   17694720  [8.8125, -1.140625, 1.015625, -1.3984375, -2.96875]
model.layers.29.mlp.ffn_sub_norm.weight         torch.Size([6912])         6912      [13.875, 12.25, 4.5625, 6.84375, 17.25]
model.layers.29.mlp.gate_proj.weight            torch.Size([6912, 2560])   17694720  [-1.1171875, -0.92578125, 2.90625, 1.3359375, 1.2109375]
model.layers.29.mlp.up_proj.weight              torch.Size([6912, 2560])   17694720  [1.4609375, 7.75, 0.357421875, -1.3203125, -0.99609375]
model.layers.29.post_attention_layernorm.weight torch.Size([2560])         2560      [1.265625, 1.3125, 1.2578125, 1.1015625, 1.28125]
model.layers.29.self_attn.attn_sub_norm.weight  torch.Size([2560])         2560      [-14.0, 13.0, 11.0625, 12.1875, -7.869675755500793e-08]
model.layers.29.self_attn.k_proj.weight         torch.Size([640, 2560])    1638400   [0.384765625, -0.470703125, -4.125, 1.0625, -0.359375]
model.layers.29.self_attn.o_proj.weight         torch.Size([2560, 2560])   6553600   [0.49609375, -1.6796875, -1.59375, -0.173828125, 5.401670932769775e-07]
model.layers.29.self_attn.q_proj.weight         torch.Size([2560, 2560])   6553600   [0.57421875, -1.125, 0.5234375, -0.5703125, 0.74609375]
model.layers.29.self_attn.v_proj.weight         torch.Size([640, 2560])    1638400   [-1.1484375, -1.15625, 4.25, 0.416015625, -1.28125]
model.norm.weight                               torch.Size([2560])         2560      [0.10302734375, 0.1005859375, 0.10205078125, 0.16015625, 0.09228515625]
```

# Todo
- Correct BitNet
  - Use llama 3 
    - Like llama, bitnet uses RMSNorm, SwiGLU, rotary embedding, and removes all biases
    - Replace all nn.Linear in attention and SwiGLU with BitLinear
    - Remove RMSNorm before attention and SwiGLU because BitLinear has built-in RMSNorm
  - learning rate scheduling
    - 1.5e-3 to 8e-4, then 5e-4 to 0
  - weight decay scheduling
    - 0.1 for 50,000 steps, then 0
- Official inference and training weights for 2.4B model
  - Support for other LLaMa sizes to train
- Binary kernels (triton?):
  - ternary weight matrix–vector product into two binary matmuls plus a subtraction
  - Custom [XNOR–popcount routines](https://arxiv.org/pdf/1905.10759) replace expensive MAC units, enabling 10× throughput improvements in CPU binary matmul kernels
- Test performance against huggingface and Microsoft bitnet.cpp
- Set up custom installation script thats nice and says jax or torch and which models to run 
- Make new hardware for it (fpga)
  - https://github.com/rejunity/tiny-asic-1_58bit-matrix-mul
  - https://www.xilinx.com/publications/presentations/binary-networks-on-fpgas-sjsu-bnn-dec-2016.pdf
  - https://jaewoong.org/pubs/fpt16-accelerating-bnn.pdf
- Make 1-bit Mixture-of-Experts (MoE)