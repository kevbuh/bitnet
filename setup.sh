#!/usr/bin/env bash
set -e

python3 -m venv venv
source venv/bin/activate


pip install torch tqdm

# weights: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/blob/main/model.safetensors
if [ -d "bitnet-b1.58-2B-4T" ]; then
  echo "Weights already downloaded."
else
  echo "Downloading weights..."
  git lfs install
  git clone https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
fi

echo "✅ Virtual environment created and dependencies installed."
echo "To activate it later, run: source venv/bin/activate"
