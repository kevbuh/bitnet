#!/usr/bin/env python3
"""
$ python bitnet/train.py --dataset custom --debug
$ python bitnet/train.py --dataset wiki --epochs 5 --batch_size 8
"""

import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from typing import Tuple, Union, List
from torch.utils.data import Dataset, DataLoader

from model import BitNet
from utils import print_model_params, training_step, calculate_model_size_in_gb, save_checkpoint, load_latest_checkpoint, validate, get_tokenizer

class TokenDataset(Dataset):
    """
    Token-level language-model dataset that returns (x, y) tensors.
    
    You can pass in either:
      - a 1D torch.Tensor of token IDs, or
      - a Python list of token IDs (it will be converted to a tensor).
    """
    def __init__(self, tokens: Union[torch.Tensor, List[int]], block_size: int):
        super().__init__()
        if isinstance(tokens, list): tokens = torch.tensor(tokens, dtype=torch.long)
        self.tokens = tokens
        self.block_size = block_size
        assert self.tokens.dim() == 1, "tokens must be a 1D sequence of IDs"
        assert len(self.tokens) > block_size, "token sequence must be longer than block_size"

    def __len__(self) -> int: return len(self.tokens) - self.block_size

    def __getitem__(self, idx: int):
        """
        Returns:
          x: LongTensor of shape (block_size,)
          y: LongTensor of shape (block_size,)
        where y[t] = x[t+1], i.e. next-token target.
        """
        chunk = self.tokens[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

tok = get_tokenizer()
def encode(text: str): return tok.encode(text, add_special_tokens=False)

def get_batch(loader_iter, loader):
    """Returns next (x, y) batch tensors, resetting iterator if exhausted."""
    try: xb, yb = next(loader_iter)
    except StopIteration:
        new_iter = iter(loader)
        xb, yb = next(new_iter)
        return xb, yb, new_iter
    return xb, yb, loader_iter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["custom", "wiki"], default="custom")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=3, help="Only used for wiki dataset")  # noqa: E501
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------- Hyper-parameters --------------------
    if args.debug: # because im gpu poor
        cfg = dict(
            batch_size     = 2,       # small so it fits in GPU memory
            block_size     = 512,     # ¼ of the full context—but long enough to catch positional bugs
            lr             = 3e-4,    # a standard debug learning‐rate
            n_embd         = 256,     # 1/10th of 2560
            n_head         = 8,       # 1/4th of 32; keeps head_dim = 256/8 = 32
            n_kv_head      = 2,       # 1/4th of 8
            ffn_dim        = 1024,    # 4× the embedding size
            n_layer        = 4,       # 1/7.5th of 30 (round to 4)
            max_iters      = 500,     # enough to see loss descend
            eval_interval  = 100,     # check validation every 100 iters
        )
    else:
        cfg = dict(batch_size=4, block_size=2048, lr=1.2e-2, n_embd=2560, n_head=32, n_kv_head=8, ffn_dim=6912, n_layer=30, max_iters=10_000, eval_interval=100) # full 2.4B bitnet model

    # allow CLI overrides
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.block_size is not None: cfg["block_size"] = args.block_size
    if args.lr is not None: cfg["lr"] = args.lr
    if args.max_iters is not None: cfg["max_iters"] = args.max_iters

    print("----- Hyper-parameters -----")
    for k, v in cfg.items():
        print(f"  {k:>13}: {v}")

    # -------------------- Load data --------------------
    if args.dataset == "custom":
        text_path = Path("data/input.txt")
        if not text_path.exists(): raise FileNotFoundError("Expected `data/input.txt` for custom dataset. Provide the file or use --dataset wiki")
        raw_text = text_path.read_text(encoding="utf-8")
        data = torch.tensor(encode(raw_text), dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
    else: # wiki
        print("Loading WikiText-2...")
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(wiki["train"]["text"])
        val_text = "\n".join(wiki["validation"]["text"])
        train_data = torch.tensor(encode(train_text), dtype=torch.long)
        val_data   = torch.tensor(encode(val_text),   dtype=torch.long)

    train_ds = TokenDataset(train_data, cfg["block_size"])
    val_ds = TokenDataset(val_data, cfg["block_size"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=True)
    
    vocab_size = tok.vocab_size
    print(f"Vocabulary size: {vocab_size}")
    print("----------------------------")

    # -------------------- Model --------------------
    model = BitNet(vocab_size=vocab_size, d_model=cfg["n_embd"], block_size=cfg["block_size"], n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_kv_head=cfg["n_kv_head"], ffn_dim=cfg["ffn_dim"]).to(device, dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    # diagnostics
    if args.debug:
        print_model_params(model)
        calculate_model_size_in_gb(model)

    # -------------------- Training loop --------------------
    best_loss = float('inf')
    start_iter, best_loss = load_latest_checkpoint(model, optimizer)
    print(f"Training for {cfg['max_iters']} iterations…")
    train_iter = iter(train_loader)
    for it in tqdm(range(start_iter, cfg["max_iters"])):
        xb, yb, train_iter = get_batch(train_iter, train_loader)
        xb, yb = xb.to(device), yb.to(device)
        if it % cfg["eval_interval"] == 0 or it == cfg["max_iters"] - 1:
            with torch.no_grad():
                logits, loss = model(xb, yb)
                val_loss = validate(model, val_loader, device)
                if args.debug: print(f"step {it}: train loss {loss:.4f}, val loss {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(model, optimizer, it, val_loss)
        training_step(model, xb, yb, optimizer)
    model.generate(cfg["block_size"], tok, device)
