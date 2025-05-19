import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils import print_model_params, training_step, print_weights, calculate_model_size_in_gb
from model import GPTLanguageModel
import os
import glob

DEBUG = True

class CharDataset(Dataset):
  def __init__(self, data, block_size):
    self.data = data
    self.block_size = block_size
  def __len__(self):
    # number of possible blocks
    return len(self.data) - self.block_size
  def __getitem__(self, idx):
    chunk = self.data[idx : idx + self.block_size + 1]
    x = torch.tensor(chunk[:-1], dtype=torch.long)
    y = torch.tensor(chunk[1:], dtype=torch.long)
    return x, y

with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Vocabulary mapping
chars = sorted(set(text))
vocab_size = len(chars)
# vocab_size = 128_256
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

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

def train():
    # ------------
    # hyperparameters
    batch_size = 4
    max_iters = 10000
    eval_interval = 100
    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # ------------
    # archiparameters
    if DEBUG: # because i'm gpu poor
        learning_rate = 3e-4
        n_embd = 1024
        n_head = 16
        n_kv_head = 4
        ffn_dim = 4096
        n_layer = 8
        block_size = 512
    else:
        learning_rate = 1.2e-2
        n_embd = 2560
        n_head = 32
        n_kv_head = 8
        ffn_dim = 6912
        n_layer = 30
        block_size = 2048
    # ------------
    train_ds = CharDataset(train_data.tolist(), block_size)
    val_ds = CharDataset(val_data.tolist(), block_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    train_iter = iter(train_loader)
    # ------------
    model = GPTLanguageModel(vocab_size=vocab_size, d_model=n_embd, block_size=block_size, n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head, ffn_dim=ffn_dim).to(device).bfloat16()
    if torch.cuda.is_available():
        model = torch.compile(model)
        training_step_fn = torch.compile(training_step)
    else: training_step_fn = training_step
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # ------------
    if DEBUG:
        print_model_params(model)
        calculate_model_size_in_gb(model)
    print(f"Training for {max_iters} iterations")

    # Load the latest checkpoint if available
    start_epoch, best_loss = load_latest_checkpoint(model, optimizer)

    for it in tqdm(range(start_epoch, max_iters)):
        try: xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb, yb = xb.to(device), yb.to(device)
        if DEBUG and (it % eval_interval == 0 or it == max_iters - 1):
            with torch.no_grad():
                logits, loss = model(xb, yb)
                val_loss = validate(model, val_loader, device)
                print(f"step {it}: train loss {loss:.4f}, validation loss {val_loss:.4f}")
                if loss < best_loss:
                    best_loss = loss
                    save_checkpoint(model, optimizer, it, loss)
        training_step_fn(model, xb, yb, optimizer)
    if DEBUG: print_weights(model)
    model.stream_output(block_size, itos, device)

if __name__ == "__main__":
    train()