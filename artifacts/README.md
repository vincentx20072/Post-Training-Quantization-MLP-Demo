# artifacts/

The trained FP32 checkpoint lives here as `fp32_model.pt`. It is **not committed** (it is a build
output); generate it reproducibly with:

```bash
python -m tinyquant.train
```

This trains the MLP for 6 epochs with fixed seeds and writes `artifacts/fp32_model.pt`
(~1 MB: the state dict plus training config). Training is deterministic, so the checkpoint is
bit-identical on every run. The benchmark, tests, and `tinyquant.inspect` all load this file.
