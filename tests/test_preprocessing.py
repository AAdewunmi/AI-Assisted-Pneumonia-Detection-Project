"""
tests/test_preprocessing.py
---------------------------
Unit test to validate preprocessing and DataLoader output.
"""

import torch
from src.data_loader import get_data_loader
from pathlib import Path
import pandas as pd


def setup_fake_data(tmp_path):
    """Create a small fake dataset for test."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)
    csv_path = tmp_path / "labels.csv"

    from PIL import Image
    import numpy as np

    # Create 5 random RGB images
    ids, labels = [], []
    for i in range(5):
        pid = f"patient_{i}"
        ids.append(pid)
        labels.append(i % 2)
        Image.fromarray((np.random.rand(256, 256, 3) * 255).astype("uint8")).save(img_dir / f"{pid}.png")

    pd.DataFrame({"patientId": ids, "Target": labels}).to_csv(csv_path, index=False)
    return csv_path, img_dir


def test_preprocessing_pipeline(tmp_path):
    """Verify loader output shapes and normalization."""
    csv_path, img_dir = setup_fake_data(tmp_path)
    loader = get_data_loader(csv_path, img_dir, batch_size=4)

    imgs, labels = next(iter(loader))

    # Assertions
    assert imgs.shape == (4, 3, 224, 224), "Unexpected image tensor shape"
    assert not torch.isnan(imgs).any(), "NaNs found in tensor"
    assert (labels.dtype == torch.long), "Labels dtype mismatch"
    assert (imgs.mean().abs() < 2), "Images not normalized properly"
