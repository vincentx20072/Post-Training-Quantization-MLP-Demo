"""Quantize a tiny tensor by hand to see every step of symmetric quantization.

    python examples/quantize_tensor.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from tinyquant.scales import compute_scale, quantization_error, quantize  # noqa: E402

w = torch.tensor([0.73, -0.18, 0.42, -0.51, 0.09, 0.88, -0.66])
bits = 8

print("FP32 values        :", [round(v, 3) for v in w.tolist()])
print("max(|W|)           :", round(w.abs().max().item(), 3))
scale = compute_scale(w, bits)
print(f"scale = max|W|/{(1 << (bits - 1)) - 1} :", round(scale.item(), 6))

qt = quantize(w, bits)
print(f"INT{bits} codes         :", qt.q.tolist())
print("reconstructed W_hat:", [round(v, 3) for v in qt.dequantize().tolist()])

err = quantization_error(w, qt)
print("abs error per value:", [round(abs(a - b), 4) for a, b in zip(w.tolist(), qt.dequantize().tolist())])
print(f"mean abs error     : {err['mean_abs_error']:.5f}   max abs error: {err['max_abs_error']:.5f}")

print("\nSame tensor at INT4 (coarser grid -> larger error):")
qt4 = quantize(w, 4)
print("INT4 codes         :", qt4.q.tolist())
print("reconstructed W_hat:", [round(v, 3) for v in qt4.dequantize().tolist()])
print(f"mean abs error     : {quantization_error(w, qt4)['mean_abs_error']:.5f}")
