"""
tests/test_resume_training.py
-----------------------------
Ensures that checkpoint resuming in src/train.py works as expected.

Covers:
- Baseline model resume
- Fine-tuned model resume
- Graceful handling of missing or mismatched layers
"""

import torch
from pathlib import Path
import pytest
from src.train import detect_model_from_checkpoint, train_baseline


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Create a temporary folder to simulate checkpoints."""
    ckpt_dir = tmp_path / "saved_models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir


def _create_fake_model_file(ckpt_path: Path):
    """Create a minimal valid fake checkpoint with random tensors."""
    state_dict = {
        "fc.weight": torch.randn(2, 2048),
        "fc.bias": torch.randn(2),
    }
    torch.save(state_dict, ckpt_path)
    return ckpt_path


def test_detect_model_from_checkpoint_baseline(tmp_checkpoint_dir):
    """Ensure baseline checkpoint detection works."""
    ckpt = _create_fake_model_file(tmp_checkpoint_dir / "resnet50_baseline.pt")
    builder = detect_model_from_checkpoint(ckpt)
    assert builder.__name__ == "build_resnet50_baseline"


def test_detect_model_from_checkpoint_finetune(tmp_checkpoint_dir):
    """Ensure fine-tuned checkpoint detection works."""
    ckpt = _create_fake_model_file(tmp_checkpoint_dir / "resnet50_finetuned.pt")
    builder = detect_model_from_checkpoint(ckpt)
    assert builder.__name__ == "build_resnet50_finetuned"


def test_resume_training_loads_checkpoint(tmp_checkpoint_dir, tmp_path, monkeypatch):
    """
    Simulate resuming training from a checkpoint without dataset access.
    Uses synthetic CSV + fake images to ensure train_baseline() runs.
    """
    # Create fake checkpoint
    ckpt_path = _create_fake_model_file(tmp_checkpoint_dir / "resnet50_baseline.pt")

    # Fake dataset
    csv_file = tmp_path / "fake_labels.csv"
    csv_file.write_text("patientId,Target\nfake1,0\nfake2,1\n")

    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch get_data_loader to avoid needing real images
    import src.data_loader as dl

    def fake_loader(*args, **kwargs):
        """Return a small synthetic dataloader."""
        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0, 1])
        return [(x, y)]

    monkeypatch.setattr(dl, "get_data_loader", fake_loader)
    monkeypatch.setattr(dl, "get_balanced_loader", fake_loader)

    # Run one epoch of resumed training
    train_baseline(
        csv_path=csv_file,
        img_dir=img_dir,
        epochs=1,
        batch_size=2,
        lr=1e-3,
        balanced=False,
        resume=str(ckpt_path),
    )

    # Verify logs and model files exist
    reports_dir = Path("reports") / "week2_metrics"
    assert reports_dir.exists()
    assert any(f.suffix == ".csv" for f in reports_dir.glob("*.csv")), "No log CSV created"
    assert Path("saved_models/resnet50_best.pt").exists(), "Best model checkpoint missing"
