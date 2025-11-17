"""
Unit tests for the balanced DataLoader and FocalLoss in PneumoDetect.
Verifies class weighting, sampling balance, and loss behavior.
"""

import torch
import pandas as pd
from pathlib import Path
from src.data_loader import get_balanced_loader, get_class_weights
from src.losses import FocalLoss


def setup_fake_labels(tmp_path):
    """Create a temporary CSV file with a known class imbalance."""
    df = pd.DataFrame({
        "patientId": [f"id_{i}" for i in range(10)],
        "Target": [0] * 8 + [1] * 2,  # 80% normal, 20% pneumonia
    })
    csv_path = tmp_path / "labels.csv"
    df.to_csv(csv_path, index=False)
    return csv_path