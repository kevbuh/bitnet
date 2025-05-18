import torch
from tqdm import tqdm
from utils import print_model_params, training_step, print_weights, calculate_model_size_in_gb
from model import GPTLanguageModel

DEBUG = True

# Data loading
with open('data/input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split, batch_size, block_size, device):
  # generate a small batch of data of inputs x and targets y
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

def train():
  # ------------
  # hyperparameters
  batch_size = 4 # how many independent sequences will we process in parallel?
  max_iters = 100
  eval_interval = 100
  
  # device = 'cpu'
  device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
  print(f"Using device: {device}")
  # ------------
  # archiparameters
  if DEBUG: # because i'm gpu poor
    learning_rate = 3e-4
    n_embd      = 1024          # hidden size
    n_head      = 16            # total attention heads
    n_kv_head   = 4             # GQA: ¼ of heads carry K & V
    head_dim    = n_embd // n_head   # 64
    ffn_dim     = 4096          # 4× d_model
    n_layer     = 8             # number of transformer blocks
    vocab_size  = 128_256       # keep full vocab, or reduce to ~32 k if you like
    block_size  = 512           # context length
  else:
    learning_rate = 1.2*10e-3
    n_embd      = 2560          # hidden size d
    n_head      = 32            # total attention heads
    n_kv_head   = 8             # GQA: heads that carry K & V
    head_dim    = n_embd // n_head # 80
    ffn_dim     = 6912
    n_layer     = 30
    vocab_size  = 128_256
    block_size  = 2048          # anything ≥ max context used in training
  # ------------
  model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, n_kv_head, ffn_dim)
  m = model.to(device).bfloat16()
  if torch.cuda.is_available(): 
    m = torch.compile(m)
    training_step_compiled = torch.compile(training_step)
  else:
    training_step_compiled = training_step
  optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
  # ------------
  if DEBUG:
    print_model_params(m)
    calculate_model_size_in_gb(model)
  print(f"Training for {max_iters} iterations")
  for iter in tqdm(range(max_iters)):
    xb, yb = get_batch('train', batch_size, block_size, device)
    if DEBUG:
      if iter % eval_interval == 0 or iter == max_iters - 1:
        with torch.no_grad():
          logits, loss = model(xb, yb)
          print(f"step {iter}: train loss {loss:.4f}")
      training_step_compiled(model, xb, yb, optimizer)
  if DEBUG: print_weights(m)
  m.stream_output(block_size, itos, device)

if __name__ == "__main__":
  train() 