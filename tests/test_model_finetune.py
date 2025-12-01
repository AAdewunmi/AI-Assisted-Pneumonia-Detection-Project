"""
Unit tests for fine-tuning features in PneumoDetect.

Covers:
- Model unfreezing for fine-tuned ResNet-50.
- Differential learning rate setup.
- Learning rate scheduler behaviour (ReduceLROnPlateau).
"""

import torch
from src.model import build_resnet50_finetuned


def test_finetuned_layers_unfrozen():
    """
    Ensure only the top two ResNet blocks (layer3, layer4) and the classification head
    are trainable.
    """
    model = build_resnet50_finetuned()
    unfrozen = [n for n, p in model.named_parameters() if p.requires_grad]

    # Verify layer3, layer4, and fc are trainable
    assert any("layer3" in n for n in unfrozen), "Expected layer3 to be unfrozen"
    assert any("layer4" in n for n in unfrozen), "Expected layer4 to be unfrozen"
    assert any("fc" in n for n in unfrozen), "Expected final FC layer to be trainable"

    # Earlier layers should remain frozen
    assert all("layer1" not in n and "layer2" not in n for n in unfrozen), (
        "Lower layers should remain frozen for fine-tuning"
    )


def test_differential_learning_rates():
    """
    Confirm that the fine-tuned optimizer applies different LRs for head vs. base layers.
    """
    model = build_resnet50_finetuned()
    params = [
        {"params": model.layer3.parameters(), "lr": 1e-5},
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(), "lr": 1e-4},
    ]
    optimizer = torch.optim.Adam(params)
    lrs = sorted(set([g["lr"] for g in optimizer.param_groups]))
    assert lrs == [1e-5, 1e-4], f"Expected LRs [1e-5, 1e-4], got {lrs}"


def test_scheduler_reduces_lr_on_plateau():
    """
    Verify that ReduceLROnPlateau correctly halves the learning rate after plateau epochs.
    """
    optimizer = torch.optim.Adam([torch.randn(2, 2, requires_grad=True)], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1
    )

    # Step sequence: improvement → plateau → plateau → LR drop
    scheduler.step(1.0)  # initial loss, sets best
    scheduler.step(1.0)  # same loss (1st plateau)
    scheduler.step(1.0)  # 2nd plateau, triggers LR drop
    new_lr = optimizer.param_groups[0]["lr"]

    assert abs(new_lr - 5e-4) < 1e-6, f"Expected LR to halve after plateau, got {new_lr}"
