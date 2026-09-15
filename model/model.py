"""The classifier: a small fully-connected network (MLP) trained in FP32.

    28x28 image -> flatten -> 784 -> Linear(784, 256) -> ReLU
                                  -> Linear(256, 128) -> ReLU
                                  -> Linear(128, 10)  -> class logits

Deliberately simple and readable: the point of TinyQuant is the quantization of the trained weights,
not the architecture. The three ``Linear`` layers hold all the learnable weights, and those weight
matrices are exactly what post-training quantization compresses.
"""
from __future__ import annotations

import torch
import torch.nn as nn

INPUT_DIM = 28 * 28
NUM_CLASSES = 10


class MLP(nn.Module):
    def __init__(self, hidden1: int = 256, hidden2: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(INPUT_DIM, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    @property
    def linear_layers(self) -> dict[str, nn.Linear]:
        """The weight-bearing layers, in forward order - the tensors quantization operates on."""
        return {"fc1": self.fc1, "fc2": self.fc2, "fc3": self.fc3}


def build_model(seed: int | None = None) -> MLP:
    if seed is not None:
        torch.manual_seed(seed)
    return MLP()
