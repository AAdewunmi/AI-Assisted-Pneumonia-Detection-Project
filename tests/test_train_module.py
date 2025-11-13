"""
tests/test_train_module.py
--------------------------
Basic smoke tests for train.py functions.
Ensures that data loader creation and training loop can initialize.
"""

import torch
from unittest.mock import patch
from src.train import train_baseline


@patch("src.train.PneumoniaDataset")
def test_train_initialization(mock_dataset, tmp_path):
    """Ensure train_baseline() runs 1 epoch without crashing using mocked dataset."""
    # Mock dataset returning random tensors
    mock_dataset.return_value.__len__.return_value = 4
    mock_dataset.return_value.__getitem__.side_effect = [
        (torch.randn(3, 224, 224), torch.tensor(0)),
        (torch.randn(3, 224, 224), torch.tensor(1)),
        (torch.randn(3, 224, 224), torch.tensor(0)),
        (torch.randn(3, 224, 224), torch.tensor(1)),
    ]

    # Run for 1 epoch, batch_size=2, log to temp dir
    csv_path = tmp_path / "fake.csv"
    img_dir = tmp_path
    train_baseline(str(csv_path), str(img_dir), epochs=1, batch_size=2, lr=1e-3)

    # Check that log file exists
    log_file = tmp_path.parent / "training_log.csv"
    assert log_file.exists(), "training_log.csv should be created after training"
