# TinyQuant

A small, educational project on **post-training quantization (PTQ)**: train a neural-network
classifier in FP32, quantize the trained weights to low-precision integers, and measure the
trade-off between **model size** and **accuracy** on Fashion-MNIST.

```
FASHION-MNIST → train FP32 MLP → final FP32 model
                                      │  post-training quantization
                                      ▼
                              quantized model  →  evaluate on the same test set
                                      │
                                      ▼
                    compare:  weight storage   ·   accuracy
```

> **Headline result.** **TinyQuant reduces model weight storage by at least 70% while keeping
> Fashion-MNIST test accuracy within 1 percentage point of the FP32 baseline.** All numbers below
> are produced by `python -m benchmarks.benchmark` on this machine — nothing is hard-coded.

---

## Measured results

Trained FP32 MLP (784→256→128→10), 6 epochs, seed 0. Evaluated on the full, unmodified 10,000-example
Fashion-MNIST test set. "Weight storage" = bit-packed integer weight codes + FP32 scale metadata
(FP32 = 4 bytes per parameter). Biases (1,576 B total) are kept in FP32 and reported separately.

| Model | precision | weight storage | size vs FP32 | test accuracy | Δ vs FP32 |
|---|---|---:|---:|---:|---:|
| FP32 baseline | float32 | 939,008 B | — | 88.32% | — |
| INT8 (per-channel) | 8-bit symmetric | 236,328 B | **−74.8%** (3.97×) | 88.30% | 0.02 pp |
| **TinyQuant (INT4)** | 4-bit symmetric | 117,388 B | **−87.5%** (8.00×) | 87.13% | 1.19 pp |

The INT8 configuration is the conservative option; the INT4 configuration is the aggressive,
maximum-compression setting that gives TinyQuant its 8× weight-storage reduction. Re-run the
benchmark to reproduce these numbers on your own hardware — the trained checkpoint is bit-identical
across retrains (fixed seeds), so the measurements are deterministic. Raw JSON is written to
[`benchmarks/results/`](benchmarks/results/).

### The claim, made explicit

The benchmark states the claim as two measurable obligations and checks both against every model:

```
size reduction  ≥ 70%   AND   accuracy loss ≤ 1.0 percentage point
```

`benchmarks/benchmark.py` records `size_reduction` and `accuracy_loss_pp` for each configuration so
the claim can be evaluated deterministically from real measurements.

---

## Install and run

Requires Python 3.10+, PyTorch and NumPy (no torchvision — Fashion-MNIST is downloaded directly).

```bash
pip install -e ".[dev]"

python -m tinyquant.train             # train FP32 model → artifacts/fp32_model.pt (a few minutes on CPU)
python -m benchmarks.benchmark        # FP32 vs INT8 vs INT4: storage + accuracy (add --latency)
python -m tinyquant.inspect           # see the quantization error for one layer
pytest                                # unit tests
python examples/quantize_tensor.py    # quantize a tiny tensor step by step
```

The first run downloads the four Fashion-MNIST IDX files into `data/` (~30 MB); after that the
project runs offline.

---

## A first course in quantization

### What is quantization?

Neural-network weights are just numbers. **Quantization** stores those numbers with *less numerical
precision* — fewer bits each — to make the model smaller and potentially cheaper to run. TinyQuant
uses **post-training quantization (PTQ)**: we train normally in FP32, and only *afterwards* convert
the finished weights to low-precision integers.

```
train  →  FP32 weights are updated repeatedly  →  final trained FP32 model  →  QUANTIZE
```

Quantization never changes what the model learned; it changes how precisely the learned weights are
stored. (This is different from quantization-aware training, which TinyQuant does not do.)

### What is FP32? What are INT8 / INT4?

**FP = floating point.** **FP32** is a 32-bit floating-point number — the default precision PyTorch
trains in, able to represent values very finely. **INT8** is an 8-bit integer (256 possible values);
**INT4** is a 4-bit integer (16 possible values). A weight's *precision* is how finely its value can
be represented:

```
FP32 weight  0.738192   ──quantize──▶   INT8 code  94   (+ a scale that maps codes back to floats)
```

Fewer bits ⇒ fewer distinct representable values ⇒ less storage, but a coarser approximation.

### What is a scale?

The **scale** is the bridge between the integer grid and the original floating-point range. TinyQuant
uses *symmetric* quantization: pick the largest magnitude weight, and divide the integer range evenly
across `[−max, +max]`:

```
scale = max(|W|) / q_max            q_max = 127 for INT8,  7 for INT4
q     = round(W / scale)  clipped to [−q_max, q_max]     # the stored integer
W_hat = q × scale                                        # the reconstructed weight
```

For example, with weights `[0.73, −0.18, 0.42]` and INT8, `scale ≈ 0.00575`, giving codes
`[127, −31, 73]` that reconstruct to `[0.730, −0.178, 0.420]`. Run `examples/quantize_tensor.py` to
see this generated live, and `python -m tinyquant.inspect` to see it for a real trained layer.

### What is quantization error, and why can accuracy fall?

`W_hat` is only an *approximation* of `W`: rounding each weight to the nearest grid point throws away
a little information. The gap `W − W_hat` is the **quantization error**. Every layer now computes with
slightly-off weights, and those small perturbations accumulate through the network, occasionally
flipping a prediction — so test accuracy can drop. Coarser precision (INT4 vs INT8) and coarser
scaling (one scale for a whole weight matrix vs one per output channel) both enlarge the error. You
can watch this happen with `tinyquant.inspect`: under aggressive INT4 with a single per-tensor scale,
many small weights round all the way to zero.

### Why do people quantize, and what is the trade-off?

Quantization shrinks the memory and storage a model needs and can make inference faster and cheaper —
which matters enormously when deploying to phones, edge devices, or at large scale. The cost is
potential accuracy loss:

```
precision ↓   ⇒   size ↓ , memory ↓ , potential efficiency ↑   ⇒   possible accuracy ↓
```

The whole engineering question is *how much* precision you can give up before the accuracy cost
outweighs the size benefit — which is exactly what TinyQuant lets you measure.

---

## What is held constant (apples-to-apples)

Every model in the benchmark shares:

```
same trained model    same test dataset    same preprocessing
same labels           same evaluation code    same classification task
```

The **only** substantive difference between the compared models is `FP32 weights` vs `quantized
weights`. That is what makes the size/accuracy comparison fair.

---

## Project layout

```
tinyquant/
├── model.py        # the FP32 MLP (784→256→128→10)
├── train.py        # train + save the reproducible checkpoint; load_model()
├── data.py         # Fashion-MNIST download + preprocessing (numpy IDX reader, no torchvision)
├── scales.py       # symmetric quantization math: scale, quantize, dequantize, error
├── quantize.py     # post-training quantization of a trained model (int8, aggressive) + storage accounting
├── evaluate.py     # deterministic top-1 accuracy on the test set
├── metrics.py      # accuracy / percentage-point / size-reduction helpers
└── inspect.py      # visualize one layer's quantization error

benchmarks/benchmark.py     tests/     examples/quantize_tensor.py     artifacts/
```

## License

MIT — see [LICENSE](LICENSE).

---

> **Evaluation fixture.** TinyQuant is a synthetic ML project created to evaluate automated benchmark
> auditing. Its implementation and measurements are real and executable; the benchmark claim is
> intentionally constructed for evaluation purposes.
