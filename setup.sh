python3 -m venv venv
source venv/bin/activate

pip install torch tqdm datasets pytest
pip install git+https://github.com/huggingface/transformers.git@096f25ae1f501a084d8ff2dcaf25fbc2bd60eba4

# weights: https://huggingface.co/microsoft/bitnet-b1.58-2B-4T/blob/main/model.safetensors
if [ -d "bitnet-b1.58-2B-4T" ]; then
  echo "Weights already downloaded."
else
  echo "Downloading weights..."
  git lfs install
  git clone https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16
