"""Quantized-model behaviour: storage really shrinks, evaluation is apples-to-apples, trade-off is real."""
import torch

from tinyquant.evaluate import evaluate
from tinyquant.metrics import percentage_points, size_reduction
from tinyquant.model import build_model
from tinyquant.quantize import CONFIGS, QuantizedModel, fp32_weight_storage_bytes


def test_configs_present():
    assert CONFIGS["int8"] == (8, 0)
    assert CONFIGS["aggressive"] == (4, None)


def test_quantized_weight_storage_is_smaller():
    model = build_model(seed=0)
    fp32 = fp32_weight_storage_bytes(model)
    for cfg in CONFIGS:
        qm = QuantizedModel.from_model(model, cfg)
        assert qm.weight_storage_bytes() < fp32
    # INT8 ~ 1/4 of FP32, aggressive INT4 ~ 1/8 (plus small scale metadata).
    assert size_reduction(fp32, QuantizedModel.from_model(model, "int8").weight_storage_bytes()) >= 0.70
    assert size_reduction(fp32, QuantizedModel.from_model(model, "aggressive").weight_storage_bytes()) >= 0.80


def test_dequantized_model_has_same_architecture(trained_model):
    qm = QuantizedModel.from_model(trained_model, "aggressive")
    deq = qm.to_dequantized_model()
    assert list(deq.linear_layers) == list(trained_model.linear_layers)
    for name, layer in deq.linear_layers.items():
        assert layer.weight.shape == trained_model.linear_layers[name].weight.shape


def test_evaluation_uses_same_test_set_and_is_deterministic(trained_model):
    a = evaluate(trained_model)
    b = evaluate(trained_model)
    assert a == b
    assert a["total"] == 10000  # full, unmodified Fashion-MNIST test set


def test_quantization_changes_only_weights_not_biases(trained_model):
    qm = QuantizedModel.from_model(trained_model, "aggressive")
    deq = qm.to_dequantized_model()
    for name, layer in deq.linear_layers.items():
        assert torch.equal(layer.bias, trained_model.linear_layers[name].bias)  # biases untouched


def test_trade_off_direction_is_real(trained_model):
    """INT8 should stay close to FP32; aggressive INT4 should lose more accuracy - from real error."""
    base = evaluate(trained_model)["accuracy"]
    int8 = evaluate(QuantizedModel.from_model(trained_model, "int8").to_dequantized_model())["accuracy"]
    aggr = evaluate(QuantizedModel.from_model(trained_model, "aggressive").to_dequantized_model())["accuracy"]
    assert percentage_points(base, int8) < percentage_points(base, aggr)
    assert aggr < base  # aggressive quantization genuinely reduces accuracy
