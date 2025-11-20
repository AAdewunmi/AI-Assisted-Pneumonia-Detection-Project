"""
Unit tests for Grad-CAM refinement (Thu - W2-D4)
------------------------------------------------
Ensures generate_cam() produces normalized, non-empty heatmaps.
"""

from pathlib import Path
import numpy as np
import torch
import pytest
from src.gradcam import generate_cam


def test_generate_cam_returns_valid_heatmap(tmp_path):
    """Ensure generate_cam() returns a properly normalized non-empty heatmap."""
    # Create dummy model checkpoint
    model_path = tmp_path / "dummy_model.pt"
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 2),
    )
    torch.save(model.state_dict(), model_path)

    # Create synthetic image
    image_path = tmp_path / "fake_image.png"
    import PIL.Image as Image

    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(image_path)

    # Run Grad-CAM generation
    heatmap = generate_cam(image_path, model_path)

    # Assertions
    assert isinstance(heatmap, np.ndarray), "Grad-CAM output must be a numpy array"
    assert heatmap.shape[-1] > 0, "Heatmap should have non-zero dimensions"
    assert np.min(heatmap) >= 0.0, f"Heatmap min value out of range: {np.min(heatmap)}"
    assert np.max(heatmap) <= 1.0, f"Heatmap max value out of range: {np.max(heatmap)}"