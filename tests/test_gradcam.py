"""
Integration tests for GradCAM on a ResNet backbone.
"""

import torch
from torchvision import models
from src.gradcam import GradCAM


def test_gradcam_generate_heatmap(tmp_path):
    """Check GradCAM produces a valid heatmap for a dummy input."""
    model = models.resnet50(weights=None)
    gradcam = GradCAM(model, target_layer_name="layer4")

    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = gradcam.generate(dummy_input, target_class=0)

    assert isinstance(heatmap, torch.Tensor)
    assert heatmap.ndim == 2
    assert 0.0 <= float(heatmap.min()) and float(heatmap.max()) <= 1.0
