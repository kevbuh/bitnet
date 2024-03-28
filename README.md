# bitnet - WIP!

![Paper diagram](https://github.com/kevbuh/bitnet/blob/main/bitimg.png)

Based on Microsoft's ['The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits'](https://arxiv.org/abs/2402.17764) paper.

This repository introduces a toy work-in-progress implementation of BitNet - a scalable and stable 1-bit Transformer architecture designed specifically for large language models.

# Key Features
BitLinear: BitNet introduces BitLinear, a drop-in replacement for the nn.Linear layer in PyTorch. BitLinear allows training of 1-bit weights from scratch, enabling efficient and accurate models.

Competitive Performance: Experimental results on language modeling tasks demonstrate that BitNet achieves competitive performance compared to state-of-the-art 8-bit quantization methods and FP16 Transformer baselines.

Memory and Energy Efficiency: By utilizing 1-bit weights and activations, BitNet substantially reduces memory footprint and energy consumption, making it an attractive solution for deploying large language models.

Scaling Capabilities: BitNet exhibits a scaling law akin to full-precision Transformers, suggesting its potential for effective scaling to even larger language models while maintaining efficiency and performance benefits.


NO MORE FLOATS!!! A very simple transformer whose weights are just [1,0,-1]


# Example

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
```


Vanilla GPT implementation is from [Andrej Karpathy](https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py)

<img src="https://i.kym-cdn.com/entries/icons/facebook/000/035/960/Screen_Shot_2020-12-02_at_3.12.45_PM.jpg" width="400">

This image represents the network well lol
