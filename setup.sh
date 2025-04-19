#!/usr/bin/env bash
set -e

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip
pip install datasets transformers tiktoken jax

# weights: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/blob/main/model.safetensors

echo "✅ Virtual environment created and dependencies installed."
echo "To activate it later, run: source venv/bin/activate"
