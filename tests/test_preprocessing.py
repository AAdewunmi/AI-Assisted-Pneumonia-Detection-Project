"""
tests/test_preprocessing.py
---------------------------
Validates preprocessing outputs and normalization consistency.
"""

import torch
from src.data_loader import get_data_loader
from pathlib import Path
import pandas as pd
import numpy as np
import cv2


def setup_fake_data(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    csv_path = tmp_path / "labels.csv"

    # create 5 fake grayscale 256x256 images
    data = []
    for i in range(5):
        pid = f"fake_{i}"
        img = (np.random.rand(256, 256) * 255).astype(np.uint8)
        cv2.imwrite(str(img_dir / f"{pid}.png"), img)
        data.append({"patientId": pid, "Target": int(i % 2)})

    pd.DataFrame(data).to_csv(csv_path, index=False)
    return csv_path, img_dir


def test_preprocessing_pipeline(tmp_path):
    csv_path, img_dir = setup_fake_data(tmp_path)
    loader = get_data_loader(csv_path, img_dir, batch_size=4)

    imgs, labels = next(iter(loader))
    assert imgs.shape == (4, 3, 224, 224), f"Unexpected image tensor shape {imgs.shape}"
    assert not torch.isnan(imgs).any(), "Image tensor should not contain NaNs"
