"""
TinyQuant benchmark: FP32 vs INT8 vs INT4 configuration.

Commands:
python -m benchmarks.benchmark
python -m benchmarks.benchmark --latency        # also time inference (optional)
"""

import argparse
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from model.data import test_loader
from model.evaluate import evaluate
from model.metrics import percentage_points, size_reduction
from model.model import MLP
from model.quantize import CONFIGS, QuantizedModel, fp32_bias_storage_bytes, fp32_weight_storage_bytes
from model.train import load_model

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def measure_latency(model, loader, warmup=1, runs=3):
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


def main():
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

    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(), "torch": torch.__version__,
        "dataset": "Fashion-MNIST", "test_examples": results["fp32"].get("accuracy") is not None and 10000,
    }
    payload = {"environment": env, "results": results}

    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "latency"} for k, v in results.items()}, indent=2))
   
    for name, r in results.items():
        if name == "fp32":
            continue

        print(
            f"  {name:11} size -{r['size_reduction']:.1%}, "
            f"accuracy loss {r['accuracy_loss_pp']:.2f} pp"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}-{args.tag}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
