"""Train the FP32 model and save the checkpoint.

Training only exists to produce a genuine trained model for quantization to act on; it is not the
object of study. A few epochs of Adam on the MLP reach ~88-89% test accuracy on Fashion-MNIST,
which is plenty to make the quantization trade-off visible. Seeds are fixed so the checkpoint is
reproducible.

    python -m model.train                      # 5 epochs, saves artifacts/fp32_model.pt
    python -m model.train --epochs 8
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from model.data import data_loaders
from model.evaluate import evaluate
from model.model import build_model

CHECKPOINT = Path(__file__).resolve().parents[1] / "artifacts" / "fp32_model.pt"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train(epochs: int = 5, lr: float = 1e-3, seed: int = 0, batch_size: int = 128, out: Path = CHECKPOINT) -> dict:
    set_seed(seed)
    train_loader, test_loader_ = data_loaders(batch_size=batch_size)
    model = build_model(seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
        acc = evaluate(model, test_loader_)["accuracy"]
        print(f"epoch {epoch}/{epochs}  train_loss {running / len(train_loader.dataset):.4f}  test_acc {acc:.4f}")

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": {"epochs": epochs, "lr": lr, "seed": seed}}, out)
    final_acc = evaluate(model, test_loader_)["accuracy"]
    print(f"saved {out}  (test accuracy {final_acc:.4f})")
    return {"path": str(out), "accuracy": final_acc}


def load_model(path: Path = CHECKPOINT) -> "torch.nn.Module":
    from model.model import MLP

    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}. Run `python -m model.train` first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = MLP()
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    train(epochs=args.epochs, lr=args.lr, seed=args.seed, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
