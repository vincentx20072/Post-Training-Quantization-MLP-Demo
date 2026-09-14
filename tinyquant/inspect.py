"""Make quantization error visible for one layer.

    python -m tinyquant.inspect                 # inspect fc1 under the aggressive config
    python -m tinyquant.inspect --layer fc2 --config int8

Shows, for the chosen layer and configuration: the FP32 weight range, the scale(s), the integer
range actually used, a few example weights alongside their reconstructed values, and the mean/max
absolute reconstruction error. This is where the trade-off becomes concrete: coarser precision means
a larger gap between the original weight and the value the model actually computes with.
"""
from __future__ import annotations

import argparse

import torch

from tinyquant.quantize import CONFIGS
from tinyquant.scales import quantization_error, quantize
from tinyquant.train import load_model


def inspect_layer(layer_name: str = "fc1", config: str = "aggressive", n_examples: int = 8) -> dict:
    model = load_model()
    layers = model.linear_layers
    if layer_name not in layers:
        raise ValueError(f"unknown layer {layer_name!r}; choose from {list(layers)}")
    bits, axis = CONFIGS[config]
    w = layers[layer_name].weight.detach()
    qt = quantize(w, bits=bits, axis=axis)
    what = qt.dequantize()
    err = quantization_error(w, qt)

    flat_w = w.flatten()
    flat_q = qt.q.flatten()
    flat_hat = what.flatten()
    idx = torch.linspace(0, flat_w.numel() - 1, n_examples).long()
    examples = [(flat_w[i].item(), int(flat_q[i].item()), flat_hat[i].item()) for i in idx]

    print(f"layer {layer_name}  config {config}  ({bits}-bit, {'per-channel' if axis == 0 else 'per-tensor'})")
    print(f"  FP32 weight range : [{w.min().item():+.5f}, {w.max().item():+.5f}]  max|W| = {w.abs().max().item():.5f}")
    scale_desc = f"{qt.scale.item():.6g}" if qt.scale.numel() == 1 else f"{qt.scale.numel()} per-channel scales in [{qt.scale.min():.3g}, {qt.scale.max():.3g}]"
    print(f"  scale             : {scale_desc}")
    print(f"  integer range used: [{int(qt.q.min())}, {int(qt.q.max())}]  (max representable +/-{(1 << (bits - 1)) - 1})")
    print(f"  examples  FP32        -> INT{bits}  -> reconstructed   (abs error)")
    for orig, q, hat in examples:
        print(f"            {orig:+.5f}  -> {q:4d}   -> {hat:+.5f}     ({abs(orig - hat):.5f})")
    print(f"  reconstruction error: mean {err['mean_abs_error']:.5f}   max {err['max_abs_error']:.5f}   rms {err['rms_error']:.5f}")
    return {"layer": layer_name, "config": config, "bits": bits, "error": err, "examples": examples}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", default="fc1", help="fc1 | fc2 | fc3")
    ap.add_argument("--config", default="aggressive", choices=list(CONFIGS))
    ap.add_argument("--examples", type=int, default=8)
    args = ap.parse_args()
    inspect_layer(args.layer, args.config, args.examples)


if __name__ == "__main__":
    main()
