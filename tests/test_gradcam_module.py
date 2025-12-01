"""
Unit tests for src/gradcam.py
Verifies GradCAM hook registration, heatmap generation, and overlay saving.
"""

import torch
import pytest
from src.gradcam import GradCAM


@pytest.fixture
def dummy_model():
    """Create a lightweight CNN to use with GradCAM."""
    import torch.nn as nn

    class TinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(8, 2)

        def forward(self, x):
            x = self.relu(self.conv(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    return TinyCNN()


def test_gradcam_hook_registration(dummy_model):
    """GradCAM should successfully register hooks on the target conv layer."""
    cam = GradCAM(dummy_model, target_layer_name="conv")
    assert hasattr(cam, "activations")
    assert hasattr(cam, "gradients")


def test_gradcam_generates_heatmap(dummy_model):
    """Ensure GradCAM returns a valid normalized heatmap tensor."""
    cam = GradCAM(dummy_model, target_layer_name="conv")
    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = cam.generate(dummy_input)
    assert isinstance(heatmap, torch.Tensor)
    assert heatmap.ndim == 2
    assert 0.0 <= float(heatmap.min()) and float(heatmap.max()) <= 1.0


def test_gradcam_saves_overlays(tmp_path, dummy_model):
    """Run GradCAM overlay generation and ensure PNGs are written to disk."""
    cam = GradCAM(dummy_model, target_layer_name="conv")
    dummy_input = torch.randn(1, 3, 224, 224)
    heatmap = cam.generate(dummy_input)

    import cv2
    import numpy as np

    img = (dummy_input.squeeze().permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    overlay = GradCAM.overlay_heatmap(img, heatmap)

    img_path = tmp_path / "sample.png"
    overlay_path = tmp_path / "sample_overlay.png"
    cv2.imwrite(str(img_path), img)
    cv2.imwrite(str(overlay_path), overlay)

    assert img_path.exists()
    assert overlay_path.exists()
