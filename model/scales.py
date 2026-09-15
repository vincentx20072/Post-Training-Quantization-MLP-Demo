"""The arithmetic of symmetric quantization: scales, rounding, clipping, reconstruction.

A weight tensor holds continuous FP32 numbers. Symmetric integer quantization maps them onto a
small, evenly spaced grid of integers centered on zero:

    scale = max(|W|) / q_max                 # one FP32 step per integer step
    q     = clip(round(W / scale), -q_max, q_max)     # the stored integer
    W_hat = q * scale                        # the reconstructed approximation

``q_max`` is the largest positive integer the target precision allows: 127 for signed INT8,
7 for signed INT4. The gap ``W - W_hat`` is the *quantization error* - the information thrown away
by rounding each weight to the nearest grid point. Fewer bits means a coarser grid means larger
error. This module is pure math with no model or PyTorch-layer knowledge, so it is easy to test.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def qmax_for_bits(bits: int) -> int:
    """Largest positive integer for signed ``bits``-bit symmetric quantization (e.g. 8 -> 127)."""
    if bits < 2:
        raise ValueError("bits must be >= 2")
    return (1 << (bits - 1)) - 1


@dataclass
class QuantizedTensor:
    """A quantized tensor: integer codes plus the FP32 scale(s) needed to reconstruct it."""
    q: torch.Tensor          # integer codes, dtype int8/int16, in [-qmax, qmax]
    scale: torch.Tensor      # per-tensor scalar, or per-channel vector (one scale per row)
    bits: int
    axis: int | None = None  # None = per-tensor; 0 = per-output-channel (per row of a weight matrix)

    def dequantize(self) -> torch.Tensor:
        """Reconstruct the approximate FP32 tensor ``q * scale``."""
        if self.axis is None:
            return self.q.to(torch.float32) * self.scale
        shape = [1] * self.q.dim()
        shape[self.axis] = -1
        return self.q.to(torch.float32) * self.scale.reshape(shape)


def compute_scale(w: torch.Tensor, bits: int, axis: int | None = None, eps: float = 1e-12) -> torch.Tensor:
    """Symmetric ``max(|W|)/q_max`` scale, per-tensor (``axis=None``) or per-channel (``axis=0``)."""
    qmax = qmax_for_bits(bits)
    if axis is None:
        amax = w.abs().max()
    else:
        amax = w.abs().amax(dim=[d for d in range(w.dim()) if d != axis])
    return (amax / qmax).clamp(min=eps)


def quantize(w: torch.Tensor, bits: int, axis: int | None = None) -> QuantizedTensor:
    """Quantize ``w`` to signed ``bits``-bit integers with a symmetric scale."""
    qmax = qmax_for_bits(bits)
    scale = compute_scale(w, bits, axis)
    if axis is None:
        divisor = scale
    else:
        shape = [1] * w.dim()
        shape[axis] = -1
        divisor = scale.reshape(shape)
    q = torch.round(w / divisor).clamp(-qmax, qmax)
    dtype = torch.int8 if bits <= 8 else torch.int16
    return QuantizedTensor(q=q.to(dtype), scale=scale, bits=bits, axis=axis)


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    return qt.dequantize()


def quantization_error(w: torch.Tensor, qt: QuantizedTensor) -> dict[str, float]:
    """Absolute reconstruction error statistics between ``w`` and its dequantized form."""
    err = (w - qt.dequantize()).abs()
    denom = w.abs().mean().clamp(min=1e-12)
    return {
        "mean_abs_error": err.mean().item(),
        "max_abs_error": err.max().item(),
        "rms_error": err.pow(2).mean().sqrt().item(),
        "mean_rel_error": (err.mean() / denom).item(),
    }
