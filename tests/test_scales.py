"""Quantization mathematics: scale, rounding, clipping, reconstruction."""
import torch

from tinyquant.scales import compute_scale, qmax_for_bits, quantization_error, quantize


def test_qmax():
    assert qmax_for_bits(8) == 127
    assert qmax_for_bits(4) == 7
    assert qmax_for_bits(2) == 1


def test_scale_is_maxabs_over_qmax():
    w = torch.tensor([0.5, -1.0, 0.25])
    assert torch.isclose(compute_scale(w, 8), torch.tensor(1.0 / 127))
    assert torch.isclose(compute_scale(w, 4), torch.tensor(1.0 / 7))


def test_quantize_rounds_and_reconstructs():
    w = torch.tensor([0.73, -0.18, 0.42])
    qt = quantize(w, 8)
    # max|W| = 0.73, scale = 0.73/127; the max-magnitude weight maps to +/-127.
    assert qt.q.max().item() == 127
    recon = qt.dequantize()
    assert torch.allclose(recon, w, atol=qt.scale.item())  # error is bounded by one scale step


def test_clipping_bounds_codes():
    w = torch.linspace(-2, 2, 50)
    for bits in (8, 4, 2):
        qt = quantize(w, bits)
        qmax = qmax_for_bits(bits)
        assert qt.q.min().item() >= -qmax and qt.q.max().item() <= qmax


def test_fewer_bits_means_more_error():
    torch.manual_seed(0)
    w = torch.randn(1000)
    e8 = quantization_error(w, quantize(w, 8))["rms_error"]
    e4 = quantization_error(w, quantize(w, 4))["rms_error"]
    e2 = quantization_error(w, quantize(w, 2))["rms_error"]
    assert e8 < e4 < e2


def test_per_channel_not_worse_than_per_tensor():
    torch.manual_seed(0)
    w = torch.randn(16, 32) * torch.tensor([[1.0]] * 8 + [[10.0]] * 8)  # rows with very different ranges
    per_tensor = quantization_error(w, quantize(w, 4, axis=None))["rms_error"]
    per_channel = quantization_error(w, quantize(w, 4, axis=0))["rms_error"]
    assert per_channel <= per_tensor


def test_dequantize_is_deterministic():
    torch.manual_seed(1)
    w = torch.randn(64, 64)
    a = quantize(w, 4).dequantize()
    b = quantize(w, 4).dequantize()
    assert torch.equal(a, b)
