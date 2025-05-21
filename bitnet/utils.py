import os
import glob
import torch
import time

def print_model_params(m):
  num_params = sum(p.numel() for p in m.parameters())
  if num_params >= 1e9: print(num_params / 1e9, 'B parameters')
  else: print(num_params / 1e6, 'M parameters')

def timeit():
  """Decorator factory to time a function's execution."""
  def decorator(func):
    def wrapper(*args, **kwargs):
      start_time = time.time()
      result = func(*args, **kwargs)
      end_time = time.time()
      step_time = end_time - start_time
      print(f"TIMEIT: {step_time:.4f} seconds")
      return result
    return wrapper
  return decorator

@timeit()
def training_step(model, xb, yb, optimizer):
  # evaluate the loss
  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  return loss

def print_weights(model):
  for name, param in model.named_parameters():
    print(f"Layer: {name:<20} Shape: {str(param.shape):<20} Stats: min={param.data.min().item():<10.4f} max={param.data.max().item():<10.4f} mean={param.data.mean().item():<10.4f} Weights: {str(param.data.flatten()[:5]):<30}...")

def calculate_model_size_in_gb(model):
  total_params = sum(p.numel() for p in model.parameters())
  # Assuming float32, which is 4 bytes per parameter
  total_size_bytes = total_params * 4
  total_size_gb = total_size_bytes / (1024 ** 3)
  print(f"Model size: {total_size_gb:.2f} GB")
  return total_size_gb 

# Function to save model checkpoint
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

# Function to load the latest checkpoint
def load_latest_checkpoint(model, optimizer, checkpoint_dir='checkpoints'):
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
    if not checkpoint_files:
        print("No checkpoints found.")
        return 0, float('inf')
    latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Loaded checkpoint '{latest_checkpoint}' (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        i = 0
        for xb, yb in val_loader:
            i += 1
            if i > 20: break
            xb, yb = xb.to(device), yb.to(device)
            logits, loss = model(xb, yb)
            total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / i
    return avg_loss

# MODEL_PARAMS = {
#   '700M': dict(hidden_size=1536, glu_size=4096, n_heads=24, n_layers=24, lr_start=1.5*10e-3, lr_end=1*10e-3),
#   '1.3B': dict(hidden_size=2048, glu_size=5460, n_heads=32, n_layers=24, lr_start=1.2*10e-3, lr_end=8*10e-4),
#   '3B': dict(hidden_size=3200, glu_size=8640, n_heads=32, n_layers=26, lr_start=1.2*10e-3, lr_end=8*10e-4),
#   '3.9B': dict(hidden_size=3200, glu_size=12800, n_heads=32, n_layers=26, lr_start=1.2*10e-3, lr_end=8*10e-4),
# }

# def get_model_params(model_size):
#     if model_size not in MODEL_PARAMS: raise ValueError(f"Model size {model_size} not found in MODEL_PARAMS")
#     hyperparams = {
#         'BATCH_SIZE': 100_000_000,
#         'NUM_TOKENS': 100_000_000_000,
#         'SEQ_LEN': 2048,
#         'WARMUP': 375,
#         'ADAM_BETA': (0.9, 0.95)
#     }
#     return MODEL_PARAMS[model_size] | hyperparams

# print(get_model_params('700M'))