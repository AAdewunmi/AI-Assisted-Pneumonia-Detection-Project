"""
tests/test_train_module.py
--------------------------
Ensures that train_baseline runs without crashing and produces expected artifacts.
"""

import torch
from unittest.mock import patch
from src.train import train_baseline
from src.data_loader import get_data_loader
from pathlib import Path
from src.train import train_baseline


def test_loader_shapes(fake_dataset):
    csv_path, img_dir = fake_dataset
    loader = get_data_loader(csv_path, img_dir, batch_size=4)
    imgs, labels = next(iter(loader))
    assert imgs.shape == (4, 3, 224, 224)


@patch("src.data_loader.PneumoniaDataset")
def test_train_initialization(mock_dataset, tmp_path):
    """
    Smoke test to ensure train_baseline() runs end-to-end without crashing
    and creates a valid log file at project root or reports directory.
    """
    mock_dataset.return_value.__len__.return_value = 4
    mock_dataset.return_value.__getitem__.side_effect = [
        (torch.randn(3, 224, 224), torch.tensor(0)),
        (torch.randn(3, 224, 224), torch.tensor(1)),
        (torch.randn(3, 224, 224), torch.tensor(0)),
        (torch.randn(3, 224, 224), torch.tensor(1)),
    ]

    csv_path = tmp_path / "fake.csv"
    img_dir = tmp_path

    train_baseline(str(csv_path), str(img_dir), epochs=1, batch_size=2, lr=1e-3)

    project_root = Path.cwd()
    reports_dir = project_root / "reports" / "week2_metrics"

    legacy_log = project_root / "training_summary.csv"
    new_logs = list(reports_dir.glob("training_log_baseline_*.csv"))

    assert legacy_log.exists() or len(new_logs) > 0, (
        "Expected training_summary.csv or week2_metrics/training_log_baseline_*.csv"
    )