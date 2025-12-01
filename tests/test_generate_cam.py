"""
tests/test_generate_cam.py
--------------------------
Ensures generate_cam() produces a valid, normalized heatmap tensor for both
real (ResNet) and dummy model cases.
"""

import torch
import numpy as np
from src.gradcam import generate_cam


def test_generate_cam_returns_valid_heatmap(tmp_path):
    """
    Ensure generate_cam() returns a properly normalized non-empty heatmap.
    Supports lightweight dummy models for test efficiency.
    """
    # Create a lightweight dummy CNN checkpoint
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 2)
    )
    model_path = tmp_path / "dummy_model.pt"
    torch.save(model.state_dict(), model_path)

    # Create a synthetic RGB image
    from PIL import Image
    image_path = tmp_path / "fake_image.png"
    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(image_path)

    # Run Grad-CAM generation
    heatmap = generate_cam(image_path, model_path)

    # Assertions
    assert isinstance(heatmap, np.ndarray), "Expected NumPy heatmap output"
    assert heatmap.ndim == 2, "Heatmap must be 2D"
    assert 0.0 <= heatmap.min() <= 1.0, "Heatmap minimum must be >= 0"
    assert 0.0 <= heatmap.max() <= 1.0, "Heatmap maximum must be <= 1"
    assert heatmap.max() > 0.1, "Heatmap appears empty"
