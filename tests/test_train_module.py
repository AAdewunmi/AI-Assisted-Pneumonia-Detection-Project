"""
tests/test_train_module.py
--------------------------
Ensures that train_baseline runs without crashing and produces expected artifacts.
"""

import torch
from unittest.mock import patch
from src.train import train_baseline


@patch("src.train.PneumoniaDataset")
def test_train_initialization(mock_dataset, tmp_path):
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

    # training_log.csv is written to project root, not tmp_path
    log_file = tmp_path.cwd() / "training_log.csv"
    assert log_file.exists(), "training_log.csv should be created in project root"
