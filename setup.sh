#!/usr/bin/env bash
set -e

python3 -m venv venv

source venv/bin/activate

echo "Installing tinygrad..."
pip install --upgrade pip
python3 -m pip install git+https://github.com/tinygrad/tinygrad.git

# weights: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/blob/main/model.safetensors
echo "Downloading weights..."
git lfs install
git clone https://huggingface.co/microsoft/bitnet-b1.58-2B-4T

echo "✅ Virtual environment created and dependencies installed."
echo "To activate it later, run: source venv/bin/activate"
