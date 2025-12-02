"""
Integration tests for Grad-CAM refinement overlays (Thu - W2-D4)
----------------------------------------------------------------
Checks overlay generation and file saving under
reports/week2_gradcam_refinement/.
"""

import numpy as np
import torch
import cv2
import PIL.Image as Image
from src.gradcam import GradCAM, generate_cam


def test_gradcam_overlay_saves(tmp_path):
    """End-to-end test: generate heatmap, overlay, and save to disk."""
    # Prepare dummy model
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 8, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 2),
    )

    model_path = tmp_path / "dummy_model.pt"
    torch.save(model.state_dict(), model_path)

    # Prepare synthetic input image
    img_path = tmp_path / "input.png"
    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(img_path)

    # Generate heatmap
    heatmap = generate_cam(img_path, model_path)

    # Overlay test
    img = cv2.imread(str(img_path))
    overlay = GradCAM.overlay_heatmap(img, heatmap)

    # Save output to reports directory
    output_dir = tmp_path / "reports" / "week2_gradcam_refinement"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "overlay_test.png"
    cv2.imwrite(str(out_path), overlay)

    # Assertions
    assert overlay.shape == img.shape, "Overlay dimensions mismatch"
    assert out_path.exists(), "Overlay file not saved"
    assert np.mean(overlay) > 0, "Overlay appears empty"
