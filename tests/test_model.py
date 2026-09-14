"""Model behaviour: output shape, 10 classes, checkpoint load."""
import torch

from tinyquant.model import NUM_CLASSES, build_model


def test_output_shape_and_classes():
    model = build_model(seed=0)
    out = model(torch.randn(5, 1, 28, 28))
    assert out.shape == (5, NUM_CLASSES)
    assert NUM_CLASSES == 10


def test_three_weight_bearing_layers():
    model = build_model(seed=0)
    assert list(model.linear_layers) == ["fc1", "fc2", "fc3"]
    assert model.fc1.weight.shape == (256, 784)
    assert model.fc3.weight.shape == (10, 128)


def test_build_is_seeded():
    a = build_model(seed=123).fc1.weight
    b = build_model(seed=123).fc1.weight
    assert torch.equal(a, b)


def test_checkpoint_loads_and_predicts(trained_model):
    out = trained_model(torch.randn(3, 1, 28, 28))
    assert out.shape == (3, 10)
    assert out.argmax(dim=1).shape == (3,)
