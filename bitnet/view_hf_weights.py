import re
import os
from safetensors import safe_open

file_path = 'data/model.safetensors'
with safe_open(file_path, framework="pt") as f:
  max_length = max(len(layer_name) for layer_name in f.keys())
  max_shape_length = 0
  max_num_params_length = 0
  total_num_parameters = 0

  def sort_key(layer_name):
    parts = re.split(r'(\d+)', layer_name)
    return [int(part) if part.isdigit() else part for part in parts]
  sorted_keys = sorted(f.keys(), key=sort_key)

  for layer_name in sorted_keys:
    weights = f.get_tensor(layer_name)
    max_shape_length = max(max_shape_length, len(str(weights.shape)))
    num_parameters = weights.numel()
    max_num_params_length = max(max_num_params_length, len(str(num_parameters)))
    print(f"{layer_name:<{max_length}} {str(weights.shape):<{max_shape_length}} {num_parameters:<{max_num_params_length}} {weights.flatten()[:5].tolist()}")
    total_num_parameters += num_parameters
  print(f"Total number of parameters: {total_num_parameters}") # 2_412_820_480 parameters

  file_size = os.path.getsize(file_path)
  file_size_gb = file_size / (1024 ** 3)
  print(f"Model size: {file_size_gb:.2f} GB")

# NOTE: inference code contains 'packed-weight' uint8's from https://huggingface.co/microsoft/bitnet-b1.58-2B-4T
# from safetensors import safe_open
# f = safe_open("bitnet-b1.58-2B-4T/model.safetensors","pt")
# for k in list(f.keys())[:4]:
#     t = f.get_tensor(k)
#     print(f"name={k} dtype={t.dtype} shape={t.shape} unique_values={t.unique()[:5]}")
