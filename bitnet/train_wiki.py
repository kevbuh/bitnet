#!/usr/bin/env python3
"""
Minimal BitNet trainer on WikiText-2 with shared vocab
======================================================

Usage:
  python train_bitnet_wikitext2.py        # 3 epochs default
  python train_bitnet_wikitext2.py --epochs 5 --batch_size 8

This script uses HuggingFace's `datasets` to load WikiText-2 (raw),
builds a shared char-level vocabulary, and trains a small BitNet.
"""

import argparse
import time
from typing import Tuple

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
from tqdm import tqdm

# ────────────────────────────────────────────────────────────
#  Char-level dataset with shared vocab
# ────────────────────────────────────────────────────────────
class CharDataset(Dataset):
    """Character-level LM dataset: returns (input, target) sequences."""
    def __init__(
        self,
        text: str,
        block_size: int,
        stoi: dict[str,int] | None = None,
        itos: dict[int,str] | None = None,
    ):
        super().__init__()
        self.block_size = block_size
        if stoi is None or itos is None:
            vocab = sorted(set(text))
            self.stoi = {ch: i for i, ch in enumerate(vocab)}
            self.itos = {i: ch for ch, i in self.stoi.items()}
        else:
            self.stoi = stoi
            self.itos = itos
        self.vocab_size = len(self.stoi)
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]

    def decode(self, ints):
        return "".join(self.itos[i] for i in ints)

# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────
def get_batch(dataset: CharDataset, batch_size: int, device: torch.device):
    idx = torch.randint(0, len(dataset), (batch_size,))
    x, y = zip(*(dataset[i] for i in idx))
    return torch.stack(x).to(device), torch.stack(y).to(device)


def estimate_loss(
    model,
    dataset: CharDataset,
    eval_iters: int = 20,
    batch_size: int = 16,
    device: torch.device = torch.device("cpu"),
):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch(dataset, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

# ────────────────────────────────────────────────────────────
#  Main training loop
# ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--load_model", type=str, default="bitnet_wikitext2_1_9500.pt", help="Path to the model file to load.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # ─── Load WikiText-2 and build shared vocab ─────────────────────────────
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")
    def join_lines(split: str) -> str:
        return "\n".join(line for line in raw[split]["text"] if line.strip())

    train_text = join_lines("train")
    val_text   = join_lines("validation")

    all_text = train_text + val_text
    vocab = sorted(set(all_text))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    print(f"Shared vocab size: {len(vocab)}")

    train_dataset = CharDataset(train_text, args.block_size, stoi=stoi, itos=itos)
    val_dataset   = CharDataset(val_text,   args.block_size, stoi=stoi, itos=itos)

    # ─── Model initialization ────────────────────────────────────────────────
    from model import BitNet  # your BitNet implementation

    model = BitNet(
        vocab_size=len(vocab),
        d_model=512,
        block_size=args.block_size,
        n_layer=6,
        n_head=8,
        n_kv_head=4,
        ffn_dim=2048,
    ).to(device)

    print(f"Model params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = (len(train_dataset) // args.batch_size) * args.epochs
    start_step = 0
    if args.load_model:
        model.load_state_dict(torch.load(args.load_model))
        print(f"Loaded model state from {args.load_model}")
        # Extract step from the model filename if possible
        try:
            start_step = int(args.load_model.split('_')[-1].split('.')[0])
        except ValueError:
            print("Could not extract step from model filename.")

    step = start_step

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        for _ in range(len(train_dataset) // args.batch_size):
            xb, yb = get_batch(train_dataset, args.batch_size, device)
            _, loss = model(xb, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1

            if step % 10000 == 0:
                val_loss = estimate_loss(
                    model,
                    val_dataset,
                    batch_size=args.batch_size,
                    device=device,
                )
                print(
                    f"Epoch {epoch} | step {step}/{total_steps} "
                    f"| train {loss:.3f} | val {val_loss:.3f}"
                )
                torch.save(model.state_dict(), f"bitnet_wikitext2_{epoch}_{step}.pt")

        print(f"Epoch {epoch} done in {time.time() - t0:.1f}s")

        # ─ sample text
        model.stream_output(
            max_new_tokens=200,
            itos=train_dataset.itos,
            device=device
        )

    torch.save(model.state_dict(), "bitnet_wikitext2.pt")
    print("Saved checkpoint to bitnet_wikitext2.pt")

if __name__ == "__main__":
    main()