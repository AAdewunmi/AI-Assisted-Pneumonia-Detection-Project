import numpy as np
from pathlib import Path
import cv2
import torch
import torch.nn as nn

from src.analysis_cam import generate_gradcam_overlay


class DummyModel(nn.Module):
    """
    Minimal model compatible with Grad-CAM:
    - exposes 'layer4'
    - produces logits requiring grad
    """
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(8, 2)

    def forward(self, x):
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits


def test_generate_gradcam_overlay(tmp_path):
    # Create fake image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img_path = tmp_path / "test.png"
    cv2.imwrite(str(img_path), img)

    model = DummyModel()

    out_path = generate_gradcam_overlay(model, str(img_path), out_dir=tmp_path)

    assert Path(out_path).exists()
    assert out_path.suffix == ".png"
