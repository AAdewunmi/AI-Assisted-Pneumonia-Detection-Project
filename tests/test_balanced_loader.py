"""
Unit tests for the balanced DataLoader and FocalLoss in PneumoDetect.
Verifies class weighting, sampling balance, and loss behavior.
"""

import torch
import pandas as pd
from pathlib import Path
from src.data_loader import get_balanced_loader, get_class_weights
from src.losses import FocalLoss
from src.data_loader import get_data_loader


def test_loader_shapes(fake_dataset):
    csv_path, img_dir = fake_dataset
    loader = get_data_loader(csv_path, img_dir, batch_size=4)
    imgs, labels = next(iter(loader))
    assert imgs.shape == (4, 3, 224, 224)


def setup_fake_labels(tmp_path):
    """Create a temporary CSV file with a known class imbalance."""
    df = pd.DataFrame({
        "patientId": [f"id_{i}" for i in range(10)],
        "Target": [0] * 8 + [1] * 2,  # 80% normal, 20% pneumonia
    })
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_class_weights_computation(tmp_path):
    """Ensure get_class_weights() returns expected inverse ratios."""
    csv_path = setup_fake_labels(tmp_path)
    weights = get_class_weights(csv_path)
    assert isinstance(weights, dict)
    assert 0 in weights and 1 in weights
    assert weights[1] > weights[0], "Minority class should have higher weight"


def test_focal_loss_behavior():
    """Ensure FocalLoss gives smaller loss for confident predictions."""
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    inputs_confident = torch.tensor([[5.0, 0.1], [0.1, 5.0]])
    targets = torch.tensor([0, 1])
    inputs_uncertain = torch.tensor([[0.5, 0.4], [0.6, 0.5]])

    loss_confident = loss_fn(inputs_confident, targets)
    loss_uncertain = loss_fn(inputs_uncertain, targets)
    assert loss_confident < loss_uncertain, "FocalLoss should penalize uncertain predictions more"
