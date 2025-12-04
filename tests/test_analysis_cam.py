import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn as nn

from src.analysis_cam import generate_gradcam_overlay


class DummyModel(nn.Module):
    """A tiny model with a single conv layer for Grad-CAM testing."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=3, padding=1)

    def forward(self, x):
        return torch.randn(1, 2)  # fake logits


def test_generate_gradcam_overlay(tmp_path, monkeypatch):
    # Create fake image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), img)

    # Dummy model for hook testing
    model = DummyModel()

    # Run Grad-CAM overlay generation
    out_path = generate_gradcam_overlay(model, str(img_path), out_dir=tmp_path)

    # Assertions
    assert Path(out_path).exists()
    assert out_path.suffix == ".png"
