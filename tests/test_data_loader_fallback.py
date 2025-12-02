"""
Additional coverage for src.data_loader fallback paths.
Ensures missing images produce dummy tensors and balanced loader still yields
batches.
"""

import torch
from src.data_loader import PneumoniaDataset, get_balanced_loader


def _write_labels(csv_path):
    csv_path.write_text("patientId,Target\nmissing1,0\nmissing2,1\n")
    return csv_path


def test_dataset_returns_dummy_tensor_when_image_missing(tmp_path):
    csv_path = _write_labels(tmp_path / "labels.csv")
    img_dir = tmp_path / "images_missing"  # do not create to trigger fallback

    dataset = PneumoniaDataset(csv_path, img_dir)
    img, label = dataset[0]
    assert label == 0
    assert img.shape == (3, 224, 224)
    assert torch.allclose(img, torch.zeros_like(img))


def test_balanced_loader_handles_missing_images(tmp_path):
    csv_path = _write_labels(tmp_path / "labels.csv")
    img_dir = tmp_path / "images_missing"  # do not create to trigger fallback

    loader = get_balanced_loader(csv_path, img_dir, batch_size=2)
    batch = next(iter(loader))
    imgs, labels = batch
    assert imgs.shape == (2, 3, 224, 224)
    # Labels should come from the CSV even when images are missing
    assert len(labels) == 2
    assert set(labels.tolist()).issubset({0, 1})
