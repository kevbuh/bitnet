from safetensors import safe_open
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Load the quantized sign & scale
with safe_open("bitnet-b1.58-2B-4T/model.safetensors", framework="jax") as f:
    sign  = f.get_tensor('model.layers.9.mlp.gate_proj.weight')   # shape (G, group_size)
    scale = f.get_tensor('model.layers.9.mlp.gate_proj.weight_scale')  # shape (G,) or (G,1)

# 2) Decode to float32 weights
decoded = sign.astype(np.float32) * scale.astype(np.float32)[..., None]

# 3) Summary statistics
print(f"mean: {decoded.mean():.6f}")
print(f"std:  {decoded.std():.6f}")
print(f"min:  {decoded.min():.6f}")
print(f"max:  {decoded.max():.6f}")

# 4) Print a small slice as a markdown table
df = pd.DataFrame(decoded[:5, :5])
print("\nFirst 5×5 decoded weights:\n")
print(df.to_markdown(index=False, tablefmt="github"))

# 5) Plot the full-weight histogram
plt.figure(figsize=(6,4))
plt.hist(decoded.flatten(), bins=50)
plt.title("Decoded Weight Distribution")
plt.xlabel("Weight Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
