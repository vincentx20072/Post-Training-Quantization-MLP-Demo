"""Deterministic evaluation on the standard Fashion-MNIST test set.

The exact same function evaluates the FP32 model and every quantized model, so accuracy differences
come only from the weights, never from a difference in how the models are measured.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from tinyquant.data import test_loader
from tinyquant.model import MLP


@torch.no_grad()
def evaluate(model: MLP, loader: DataLoader | None = None) -> dict[str, float | int]:
    """Top-1 accuracy over the full test set. Model is put in eval mode; no gradients."""
    model.eval()
    loader = loader or test_loader()
    correct = 0
    total = 0
    for x, y in loader:
        logits = model(x)
        correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.numel()
    return {"correct": correct, "total": total, "accuracy": correct / total}
