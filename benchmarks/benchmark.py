"""TinyQuant benchmark: FP32 vs INT8 vs the aggressive INT4 configuration.

Everything is held constant except the numeric precision of the weights - same trained checkpoint,
same Fashion-MNIST test set, same preprocessing, same evaluation code - so the measured differences
are attributable only to quantization.

For each model it reports:
  * weight_storage_bytes  - bit-packed integer codes + FP32 scale metadata (FP32 = 4 bytes/param)
  * accuracy              - top-1 on the full 10,000-example test set
  * accuracy_loss_pp      - percentage points below the FP32 baseline
  * size_reduction        - fraction smaller than FP32 weight storage
and, against the README claim (size reduction >= threshold AND accuracy loss <= threshold), whether
each model satisfies it.

    python -m benchmarks.benchmark
    python -m benchmarks.benchmark --latency        # also time inference (optional)
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from tinyquant.data import test_loader
from tinyquant.evaluate import evaluate
from tinyquant.metrics import percentage_points, size_reduction
from tinyquant.model import MLP
from tinyquant.quantize import CONFIGS, QuantizedModel, fp32_bias_storage_bytes, fp32_weight_storage_bytes
from tinyquant.train import load_model

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# The README claim, made explicit so the benchmark and any auditor evaluate the same thresholds.
CLAIM = {"min_size_reduction": 0.70, "max_accuracy_loss_pp": 1.0}


def measure_latency(model: MLP, loader, warmup: int = 1, runs: int = 3) -> dict:
    model.eval()
    batches = list(loader)
    with torch.no_grad():
        for _ in range(warmup):
            for x, _ in batches:
                model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            for x, _ in batches:
                model(x)
            times.append(time.perf_counter() - t0)
    n = sum(y.numel() for _, y in batches)
    return {"seconds_median": statistics.median(times), "images_per_second": n / statistics.median(times)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latency", action="store_true", help="also measure inference latency (optional metric)")
    ap.add_argument("--tag", default="benchmark")
    args = ap.parse_args()

    fp32 = load_model()
    loader = test_loader()
    fp32_wbytes = fp32_weight_storage_bytes(fp32)
    base_acc = evaluate(fp32, loader)["accuracy"]

    results: dict[str, dict] = {
        "fp32": {
            "precision_bits": 32, "scheme": "float32",
            "weight_storage_bytes": fp32_wbytes,
            "bias_storage_bytes": fp32_bias_storage_bytes(fp32),
            "accuracy": base_acc, "accuracy_loss_pp": 0.0, "size_reduction": 0.0,
        }
    }
    if args.latency:
        results["fp32"]["latency"] = measure_latency(fp32, loader)

    for cfg in CONFIGS:
        qm = QuantizedModel.from_model(fp32, cfg)
        acc = evaluate(qm.to_dequantized_model(), loader)["accuracy"]
        wbytes = qm.weight_storage_bytes()
        entry = {
            "precision_bits": qm.bits, "scheme": f"{qm.bits}-bit symmetric, {'per-channel' if qm.axis == 0 else 'per-tensor'}",
            "weight_storage_bytes": wbytes, "scale_metadata_bytes": qm.scale_metadata_bytes(), "bias_storage_bytes": qm.bias_storage_bytes(),
            "accuracy": acc, "accuracy_loss_pp": percentage_points(base_acc, acc), "size_reduction": size_reduction(fp32_wbytes, wbytes),
        }
        if args.latency:
            entry["latency"] = measure_latency(qm.to_dequantized_model(), loader)
        results[cfg] = entry

    # Evaluate the README claim against every model (the aggressive model is the claim's subject).
    for name, r in results.items():
        r["claim_satisfied"] = bool(r["size_reduction"] >= CLAIM["min_size_reduction"] and r["accuracy_loss_pp"] <= CLAIM["max_accuracy_loss_pp"])

    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(), "torch": torch.__version__,
        "dataset": "Fashion-MNIST", "test_examples": results["fp32"].get("accuracy") is not None and 10000,
    }
    payload = {"environment": env, "claim": CLAIM, "results": results}

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "latency"} for k, v in results.items()}, indent=2))
    print("\nClaim: weight storage reduced >= {:.0%} AND accuracy loss <= {:.1f} pp".format(CLAIM["min_size_reduction"], CLAIM["max_accuracy_loss_pp"]))
    for name, r in results.items():
        if name == "fp32":
            continue
        verdict = "SATISFIES" if r["claim_satisfied"] else "VIOLATES"
        print(f"  {name:11} size -{r['size_reduction']:.1%}, accuracy loss {r['accuracy_loss_pp']:.2f} pp  ->  {verdict}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}-{args.tag}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
