"""
tests/test_model_init.py
------------------------
Unit tests for src/model.py to ensure model builds correctly.
"""

import torch
from src.model import build_resnet50_baseline


def test_model_initialization():
    """Model should initialize and produce expected output shape."""
    model = build_resnet50_baseline(num_classes=2, freeze_backbone=True)
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    assert output.shape == (2, 2), "Output shape should be (batch_size, num_classes)"
    assert not torch.isnan(output).any(), "Model output should not contain NaN values"
