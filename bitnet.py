import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
print('------------')
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f'DEVICE: {device}')
eval_iters = 50
n_embd = 384
n_head = 4
assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
n_layer = 2
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# TODO: make BPE tokenizer
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
split_num = 0.9
n = int(split_num*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]
print(f'DATA SPLIT: {split_num}')

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    patrick.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = patrick(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    patrick.train()
    return out

class CustomLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, epsilon=1e-5):
        super(CustomLinear, self).__init__(in_features, out_features, bias)
        self.epsilon = epsilon
        # Initialize weights and biases here if you want to use custom initialization
        self.reset_parameters()  # You can override this method instead if you prefer

    def reset_parameters(self):
        super().reset_parameters()  # Call default initialization
        self.apply_custom_weight_modifications()

    def apply_custom_weight_modifications(self):
        with torch.no_grad():
            gamma = torch.abs(self.weight).mean()
            normalized_weights = self.weight / (gamma) #+ self.epsilon) # TODO: fix epsilon
            clipped_weights = torch.clamp(torch.round(normalized_weights), -1, 1)
            self.weight.copy_(clipped_weights)

    def forward(self, input):
        # Hook to clamp the weights before each forward pass
        self.weight.data = self.clamp_weights(self.weight.data)
        return super(CustomLinear, self).forward(input)

    def clamp_weights(self, weights):
        gamma = torch.abs(weights).mean()
        normalized_weights = weights / (gamma + self.epsilon)
        clipped_weights = torch.clamp(torch.round(normalized_weights), -1, 1)
        return clipped_weights
    
    def verify_clamping(self):
        # Check if any weights are outside the bounds [-1, 1]
        if not torch.all(torch.ge(self.weight, -1)) or not torch.all(torch.le(self.weight, 1)):
            # print("Warning: Weights are not correctly clamped.")
            pass
            return False
        else:
            print("All weights are correctly clamped between [-1, 1].")
            return True


# Function to print the weights of all linear layers
def print_linear_layer_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module, CustomLinear):  # Check for both in case you are using the custom class
            print(f"Weights of '{name}':\n{module.weight.data}")
            

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = CustomLinear(n_embd, head_size, bias=False)
        self.query = CustomLinear(n_embd, head_size, bias=False)
        self.value = CustomLinear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = CustomLinear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            CustomLinear(n_embd, 4 * n_embd),
            nn.ReLU(),
            CustomLinear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BitLM(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = CustomLinear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
    def verify_all_clamping(self):
        for module in self.modules():
            if isinstance(module, CustomLinear):
                if not module.verify_clamping():
                    return False
        return True

# ------------


patrick = BitLM()
m = patrick.to(device)
print(f'MODEL: {patrick.__class__.__name__}')

# print the number of parameters in the model
print(f'MODEL PARAMS: {sum(p.numel() for p in m.parameters())/1e6} M')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(patrick.parameters(), lr=learning_rate)
print(f'OPTIMIZER: {optimizer.__class__.__name__}')
print('------------')

start_iter = 0
load_train = True
if load_train:
    # load checkpoint weights
    print("loading checkpoint weights")
    checkpoint_path = 'checkpoints/checkpoint_iter_10501.pth'  # Replace with the path to your checkpoint
    checkpoint = torch.load(checkpoint_path)
    patrick.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_iter = checkpoint['iter']
print('training...')


for iter in range(max_iters):
    if start_iter:
        iter += start_iter
    else:
        iter = iter
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        checkpoint = {
        'iter': iter + 1,
        'state_dict': patrick.state_dict(),
        'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, f'checkpoints/checkpoint_iter_{iter+1}.pth')

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = patrick(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# print("WEIGHTS------")
# print_linear_layer_weights(patrick)
# print("------WEIGHTS")

# generate from the model
print("WEIGHT VERIFICATION:", patrick.verify_all_clamping())
print('GENERATING TEXT...')
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("GENERATION: ", patrick.generate(context, max_new_tokens=100)[0].tolist())
print("DECODE: ", decode(patrick.generate(context, max_new_tokens=100)[0].tolist()))


#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))
# Call the function