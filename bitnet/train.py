#!/usr/bin/env python3
"""
$ python bitnet/train.py --dataset char --debug
$ python bitnet/train.py --dataset wiki --epochs 5 --batch_size 8
"""

import torch
import math
import argparse
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

from utils import print_model_params, training_step, calculate_model_size_in_gb, save_checkpoint, load_latest_checkpoint, validate
from model import BitNet

class CharDataset(Dataset):
    """Character-level language-model dataset that returns (x, y) tensors."""
    def __init__(self, data: torch.Tensor, block_size: int):
        super().__init__()
        self.data = data
        self.block_size = block_size
    def __len__(self) -> int: return len(self.data) - self.block_size
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def build_vocab_and_data(text: str):
    """Build (stoi, itos, encode) plus tensorised data from raw text."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    encode = lambda s: [stoi[c] for c in s]
    return stoi, itos, torch.tensor(encode(text), dtype=torch.long)

def get_batch(loader_iter, loader):
    """Returns next (x, y) batch tensors, resetting iterator if exhausted."""
    try: xb, yb = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        xb, yb = next(loader_iter)
    return xb, yb, loader_iter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["char", "wiki"], default="char")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=3, help="Only used for wiki dataset")  # noqa: E501
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -------------------- Hyper-parameters --------------------
    if args.debug:
        if args.dataset == "wiki": cfg = dict(batch_size=16, block_size=192, lr=3e-4, n_embd=512, n_head=8, n_kv_head=4, ffn_dim=2048, n_layer=6, max_iters=None, eval_interval=200)
        else: cfg = dict(batch_size=4, block_size=512, lr=3e-4, n_embd=1024, n_head=16, n_kv_head=4, ffn_dim=4096, n_layer=8, max_iters=1_000, eval_interval=100)
    else:
        cfg = dict(batch_size=4, block_size=2048, lr=1.2e-2, n_embd=2560, n_head=32, n_kv_head=8, ffn_dim=6912, n_layer=30, max_iters=10_000, eval_interval=100) # full 2.4B bitnet model

    # allow CLI overrides
    if args.batch_size is not None: cfg["batch_size"] = args.batch_size
    if args.block_size is not None: cfg["block_size"] = args.block_size
    if args.lr is not None: cfg["lr"] = args.lr
    if args.max_iters is not None: cfg["max_iters"] = args.max_iters

    print("----- Hyper-parameters -----")
    for k, v in cfg.items():
        if k not in {"n_embd", "n_head", "n_kv_head", "ffn_dim", "n_layer"}:
            print(f"  {k:>12}: {v}")

    # -------------------- Load data --------------------
    if args.dataset == "char":
        text_path = Path("data/input.txt")
        if not text_path.exists():
            raise FileNotFoundError("Expected `data/input.txt` for char dataset. Provide the file or use --dataset wiki")
        raw_text = text_path.read_text(encoding="utf-8")
        stoi, itos, data = build_vocab_and_data(raw_text)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]
    else: # wiki
        print("Loading WikiText-2...")
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1")
        train_text = "\n".join(wiki["train"]["text"])
        val_text = "\n".join(wiki["validation"]["text"])
        stoi, itos, _ = build_vocab_and_data(train_text + val_text)
        encode = lambda s: [stoi[c] for c in s]
        train_data = torch.tensor(encode(train_text), dtype=torch.long)
        val_data = torch.tensor(encode(val_text), dtype=torch.long)

    train_ds = CharDataset(train_data, cfg["block_size"])
    val_ds = CharDataset(val_data, cfg["block_size"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, drop_last=True)
    
    vocab_size = len(stoi)
    print(f"Vocabulary size: {vocab_size}")
    print("----------------------------------------")

    # -------------------- Model --------------------
    model = BitNet(vocab_size=vocab_size, d_model=cfg["n_embd"], block_size=cfg["block_size"], n_layer=cfg["n_layer"], n_head=cfg["n_head"], n_kv_head=cfg["n_kv_head"], ffn_dim=cfg["ffn_dim"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])

    # diagnostics
    if args.debug:
        print_model_params(model)
        calculate_model_size_in_gb(model)

    # checkpoint
    start_iter, best_loss = load_latest_checkpoint(model, optimizer)

    # derive max_iters for wiki if not explicitly set
    if cfg["max_iters"] is None:
        steps_per_epoch = math.ceil(len(train_loader))
        cfg["max_iters"] = args.epochs * steps_per_epoch
        print(f"Computed max_iters = {cfg['max_iters']} (epochs={args.epochs}, steps/epoch={steps_per_epoch})")

    # -------------------- Training loop --------------------
    print(f"Training for {cfg['max_iters']} iterations…")
    train_iter = iter(train_loader)

    for it in tqdm(range(start_iter, cfg["max_iters"])):

        xb, yb, train_iter = get_batch(train_iter, train_loader)
        xb, yb = xb.to(device), yb.to(device)

        if args.debug and (it % cfg["eval_interval"] == 0 or it == cfg["max_iters"] - 1):
            with torch.no_grad():
                logits, loss = model(xb, yb)
                val_loss = validate(model, val_loader, device)
                print(f"step {it}: train loss {loss:.4f}, val loss {val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    save_checkpoint(model, optimizer, it, val_loss)
        training_step(model, xb, yb, optimizer)
    model.stream_output(cfg["block_size"], itos, device)