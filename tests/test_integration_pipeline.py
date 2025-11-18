"""
Integration test for PneumoDetect training pipeline.

Goal:
Ensure the model, data loader, and training loop run for 1 epoch
without crashes, and that outputs (logs + checkpoints) are created.
"""

from pathlib import Path
from src.train import train_baseline


def test_full_training_pipeline(fake_dataset, tmp_path):
    """
    End-to-end smoke test for one epoch of baseline training.
    Verifies that model files and training logs are produced.
    """
    csv_path, img_dir = fake_dataset
    cwd = Path.cwd()
    saved_models_dir = cwd / "saved_models"
    legacy_log = cwd / "training_summary.csv"
    reports_dir = cwd / "reports" / "week2_metrics"

    # Clean old artifacts
    if saved_models_dir.exists():
        for f in saved_models_dir.glob("*.pt"):
            f.unlink()
    if legacy_log.exists():
        legacy_log.unlink()
    if reports_dir.exists():
        for f in reports_dir.glob("*.csv"):
            f.unlink()

    # Run one epoch (CPU)
    train_baseline(
        csv_path=csv_path,
        img_dir=img_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        balanced=False,
    )

    # --- Assertions ---
    assert (saved_models_dir / "resnet50_baseline.pt").exists(), "Final model not saved"
    assert (saved_models_dir / "resnet50_best.pt").exists(), "Best model not saved"

    # Accept either of the valid log outputs
    has_legacy = legacy_log.exists()
    has_new = any(reports_dir.glob("training_log_baseline_*.csv"))
    assert has_legacy or has_new, (
        "Expected either training_summary.csv or training_log_baseline_*.csv, but none found."
    )
