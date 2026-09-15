"""Fashion-MNIST loading, with no torchvision dependency.

Fashion-MNIST is a drop-in replacement for MNIST: 70,000 28x28 grayscale images of clothing in 10
classes (T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot), split
into 60,000 train and 10,000 test examples. We use the standard, unmodified test set for every
evaluation in this project.

The four raw IDX files are downloaded once from the official Zalando mirror into ``data/`` and read
with numpy. Pixels are scaled to ``[0, 1]`` and standardized with the dataset's global mean/std -
the identical preprocessing is used for training and for every evaluation, so the only thing that
ever changes between the compared models is the numeric precision of the weights.
"""
from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MIRROR = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Standard Fashion-MNIST normalization constants (train-set mean/std of the [0,1] pixels).
MEAN, STD = 0.2860, 0.3530


def _download(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / name
    if not path.exists():
        url = MIRROR + name
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, path)
    return path


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        ndim = magic & 0xFF                                   # low byte of the magic = number of dimensions
        dims = struct.unpack(f">{ndim}I", f.read(4 * ndim))   # one 4-byte big-endian size per dimension
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_split(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(images, labels)`` for one split; images are standardized float32 of shape (N, 1, 28, 28)."""
    imgs = _read_idx(_download(FILES["train_images" if train else "test_images"]))
    labels = _read_idx(_download(FILES["train_labels" if train else "test_labels"]))
    x = torch.from_numpy(imgs.astype(np.float32) / 255.0).view(-1, 1, 28, 28)
    x = (x - MEAN) / STD
    y = torch.from_numpy(labels.astype(np.int64))
    return x, y


def data_loaders(batch_size: int = 128, eval_batch_size: int = 1000) -> tuple[DataLoader, DataLoader]:
    xtr, ytr = load_split(train=True)
    xte, yte = load_split(train=False)
    train = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True)
    test = DataLoader(TensorDataset(xte, yte), batch_size=eval_batch_size, shuffle=False)
    return train, test


def test_loader(eval_batch_size: int = 1000) -> DataLoader:
    xte, yte = load_split(train=False)
    return DataLoader(TensorDataset(xte, yte), batch_size=eval_batch_size, shuffle=False)
