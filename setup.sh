#!/usr/bin/env bash
set -e

python3 -m venv venv

source venv/bin/activate

pip install --upgrade pip
pip install jax jaxlib flax safetensors transformers numpy

# weights: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/blob/main/model.safetensors
git lfs install
git clone https://huggingface.co/microsoft/bitnet-b1.58-2B-4T

echo "✅ Virtual environment created and dependencies installed."
echo "To activate it later, run: source venv/bin/activate"
