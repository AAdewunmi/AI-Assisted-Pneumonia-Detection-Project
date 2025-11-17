"""
Integration test for PneumoDetect training pipeline.

Goal:
Ensure the model, data loader, and training loop run for 1 epoch
without crashes, and that outputs (logs + checkpoints) are created.
"""

import torch
from pathlib import Path
from src.train import train_baseline


def test_full_training_pipeline(fake_dataset, tmp_path):
    """
    End-to-end smoke test for one epoch of baseline training.
    """
    csv_path, img_dir = fake_dataset

    # Set output dirs relative to test temp folder
    cwd = Path.cwd()
    saved_models_dir = cwd / "saved_models"
    log_path = cwd / "training_log.csv"

    # Clean up old artifacts
    if saved_models_dir.exists():
        for f in saved_models_dir.glob("*.pt"):
            f.unlink()
    if log_path.exists():
        log_path.unlink()

    # Run one epoch (CPU only)
    train_baseline(
        csv_path=csv_path,
        img_dir=img_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        balanced=False,
    )

    # Assertions: model + log must exist
    assert (saved_models_dir / "resnet50_baseline.pt").exists(), "Final model not saved"
    assert (saved_models_dir / "resnet50_best.pt").exists(), "Best model not saved"
    assert log_path.exists(), "training_log.csv not found"

    # Validate log content
    import pandas as pd
    df = pd.read_csv(log_path)
    assert "accuracy" in df.columns, "training_log missing accuracy column"
    assert len(df) == 1, "Expected exactly 1 training epoch"

    # Confirm accuracy values are in valid range
    acc = df["accuracy"].iloc[0]
    assert 0.0 <= acc <= 1.0, f"Invalid accuracy value: {acc}"
