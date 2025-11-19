import torch
from torchvision import models
from src.gradcam import GradCAM
from PIL import Image
import numpy as np
from pathlib import Path


def test_gradcam_initialization():
    """Ensure GradCAM correctly registers hooks."""
    model = models.resnet50(weights=None)
    gradcam = GradCAM(model, target_layer_name="layer4")
    assert hasattr(gradcam, "target_layer")
    assert gradcam.target_layer is not None


# def test_gradcam_generate_heatmap(tmp_path):
#     """Check GradCAM produces a valid heatmap for a dummy input."""
#     model = models.resnet50(weights=None)
#     gradcam = GradCAM(model, target_layer_name="layer4")

#     dummy_input = torch.randn(1, 3, 224, 224)
#     heatmap = gradcam.generate(dummy_input, target_class=0)
#     assert heatmap.ndim == 2
#     assert heatmap.dtype == np.uint8
#     assert 0 <= heatmap.max() <= 255


# def test_overlay_and_save(tmp_path):
#     """Verify overlay generation and file saving."""
#     img = Image.new("RGB", (224, 224), color="gray")
#     heatmap = np.uint8(np.random.rand(224, 224) * 255)

#     model = models.resnet50(weights=None)
#     gradcam = GradCAM(model, target_layer_name="layer4")

#     overlay = gradcam.overlay_heatmap(heatmap, img)
#     assert overlay.shape == (224, 224, 3)

#     output_file = tmp_path / "overlay_test.png"
#     gradcam.save_overlay(img, heatmap, str(output_file))
#     assert output_file.exists()
