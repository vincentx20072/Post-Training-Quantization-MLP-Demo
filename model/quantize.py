"""Post-training quantization (PTQ) of a trained FP32 model.

PTQ happens *after* training has finished:

    train -> final FP32 weights -> quantize -> evaluate

Nothing is re-trained here. Each ``Linear`` layer's weight matrix is quantized with the symmetric
scheme in :mod:`model.scales`; biases (a few hundred values, negligible for storage) are kept in
FP32, which is standard for weight-only PTQ. Two named configurations are provided:

* ``int8``       - 8-bit, **per-output-channel** scales (one scale per row of the weight matrix).
                   A separate scale per channel fits each row's range tightly, so the error is small.
* ``int4``       - 4-bit, a **single per-tensor** scale for the whole weight matrix (the aggressive,
                   maximum-compression setting). Four bits give only 15 grid points, and one global
                   scale must cover every row's range at once, so weights in narrow-range rows are
                   quantized very coarsely. This is a legitimate, deliberately low-flexibility
                   configuration - no weights are corrupted; the larger error is the honest
                   consequence of fewer bits and coarser scaling.

To *evaluate* a quantized model we build the same MLP architecture and load the **dequantized**
weights (``q * scale``). The network then runs with exactly the reconstructed values the integer
codes represent - so accuracy reflects real quantization error, while everything else (architecture,
test set, preprocessing, evaluation code) is held constant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from model.model import MLP
from model.scales import QuantizedTensor, quantize

# name -> (bits, axis)  ;  axis=0 per-output-channel, None per-tensor
CONFIGS: dict[str, tuple[int, int | None]] = {
    "int8": (8, 0),          # conservative: 8-bit, per-channel
    "int4": (4, None),       # aggressive: 4-bit, single per-tensor scale
}


@dataclass
class QuantizedLayer:
    weight: QuantizedTensor
    bias: torch.Tensor | None  # kept FP32

    def dequantized_weight(self) -> torch.Tensor:
        return self.weight.dequantize()


class QuantizedModel:
    """A trained model with its weight matrices quantized under one configuration."""

    def __init__(self, config: str, bits: int, axis: int | None, layers: dict[str, QuantizedLayer], template: MLP):
        self.config = config
        self.bits = bits
        self.axis = axis
        self.layers = layers
        self._template = template

    @classmethod
    def from_model(cls, model: MLP, config: str) -> "QuantizedModel":
        if config not in CONFIGS:
            raise ValueError(f"unknown config {config!r}; choose from {list(CONFIGS)}")
        bits, axis = CONFIGS[config]
        layers: dict[str, QuantizedLayer] = {}
        for name, layer in model.linear_layers.items():
            qt = quantize(layer.weight.detach(), bits=bits, axis=axis)
            bias = layer.bias.detach().clone() if layer.bias is not None else None
            layers[name] = QuantizedLayer(weight=qt, bias=bias)
        return cls(config, bits, axis, layers, model)

    def to_dequantized_model(self) -> MLP:
        """Rebuild the MLP with reconstructed (dequantized) weights for evaluation."""
        model = MLP()
        with torch.no_grad():
            for name, layer in model.linear_layers.items():
                ql = self.layers[name]
                layer.weight.copy_(ql.dequantized_weight())
                if ql.bias is not None:
                    layer.bias.copy_(ql.bias)
        return model.eval()

    # ---- storage accounting -------------------------------------------------------------------

    def weight_storage_bytes(self) -> int:
        """Bytes to store the quantized weight *codes*, bit-packed to the target precision, plus scales."""
        total = 0
        for ql in self.layers.values():
            n = ql.weight.q.numel()
            total += math.ceil(n * self.bits / 8)                       # packed integer codes
            total += ql.weight.scale.numel() * 4                        # FP32 scale metadata
        return total

    def scale_metadata_bytes(self) -> int:
        return sum(ql.weight.scale.numel() * 4 for ql in self.layers.values())

    def bias_storage_bytes(self) -> int:
        return sum(ql.bias.numel() * 4 for ql in self.layers.values() if ql.bias is not None)

    def num_weight_params(self) -> int:
        return sum(ql.weight.q.numel() for ql in self.layers.values())


def fp32_weight_storage_bytes(model: MLP) -> int:
    """Bytes to store the FP32 weight matrices (4 bytes per parameter)."""
    return sum(layer.weight.numel() * 4 for layer in model.linear_layers.values())


def fp32_bias_storage_bytes(model: MLP) -> int:
    return sum(layer.bias.numel() * 4 for layer in model.linear_layers.values() if layer.bias is not None)
