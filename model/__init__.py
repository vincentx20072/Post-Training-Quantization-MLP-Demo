"""TinyQuant: post-training quantization of a small Fashion-MNIST classifier."""
from model.model import MLP, build_model
from model.quantize import CONFIGS, QuantizedModel, fp32_weight_storage_bytes
from model.scales import QuantizedTensor, compute_scale, dequantize, quantization_error, quantize

__version__ = "0.1.0"
__all__ = ["MLP", "build_model", "QuantizedModel", "CONFIGS", "fp32_weight_storage_bytes",
           "QuantizedTensor", "compute_scale", "quantize", "dequantize", "quantization_error"]
