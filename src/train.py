"""
Training script for PneumoDetect baseline, balanced, and fine-tuned models.

Features:
- Supports unbalanced (standard) and weighted (balanced) sampling.
- Optionally unfreezes upper ResNet blocks for fine-tuning.
- Uses Adam optimizer with differential learning rates.
- Includes ReduceLROnPlateau scheduler to adapt LR on validation loss.
- Automatically saves best and final model checkpoints.
- Logs loss, accuracy, and learning rate per epoch to CSV for analysis.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from src.data_loader import get_data_loader, get_balanced_loader, get_default_transform


def collate_skip_none(batch):
    """Remove None entries (missing images) from minibatches."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels)


def build_resnet50_baseline(num_classes: int = 2, fine_tune: bool = False) -> nn.Module:
    """
    Return a pretrained ResNet-50.
    If fine_tune=True, unfreeze top layers (layer3, layer4).
    """
    model = models.resnet50(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False

    # Optional fine-tuning
    if fine_tune:
        for layer_name in ["layer3", "layer4"]:
            for param in getattr(model, layer_name).parameters():
                param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ensure_dataset_available(csv_path: Path) -> Path:
    """
    Ensure dataset CSV exists.
    If missing, create a small synthetic CSV for testing.
    """
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Creating synthetic CSV for CI/testing.")
        tmp_dir = Path("data")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "tmp_labels.csv"
        df = pd.DataFrame(
            {"patientId": [f"fake_{i}" for i in range(10)], "Target": [0, 1] * 5}
        )
        df.to_csv(tmp_path, index=False)
        return tmp_path
    return csv_path


def train_baseline(
    csv_path,
    img_dir,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    balanced: bool = False,
    fine_tune: bool = False,
):
    """
    Train baseline, balanced, or fine-tuned ResNet-50 model.
    Saves best checkpoint and training logs with LR tracking.
    """
    csv_path = ensure_dataset_available(Path(csv_path))
    img_dir = Path(img_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = get_default_transform()
    if balanced:
        loader = get_balanced_loader(csv_path, img_dir, transform, batch_size=batch_size)
    else:
        loader = get_data_loader(csv_path, img_dir, transform, batch_size=batch_size)

    model = build_resnet50_baseline(fine_tune=fine_tune).to(device)

    # Differential learning rates for fine-tuning
    if fine_tune:
        params = [
            {"params": model.layer3.parameters(), "lr": lr / 10},
            {"params": model.layer4.parameters(), "lr": lr / 10},
            {"params": model.fc.parameters(), "lr": lr},
        ]
        optimizer = torch.optim.Adam(params)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, verbose=False
    )

    best_acc = 0.0
    logs = []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in pbar:
            if batch is None:
                continue
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}"})

        epoch_loss = running_loss / max(1, len(loader))
        epoch_acc = correct / total if total > 0 else 0.0
        scheduler.step(epoch_loss)  # adapt LR based on loss

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}, LR={current_lr:.6f}"
        )

        logs.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "lr": current_lr,
            }
        )

        # Save best checkpoint
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            Path("saved_models").mkdir(parents=True, exist_ok=True)
            best_path = Path(
                "saved_models/resnet50_finetuned.pt"
                if fine_tune
                else "saved_models/resnet50_best.pt"
            )
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (Accuracy={best_acc:.4f}) → {best_path}")

    # Write logs (with LR)
    reports_dir = Path("reports") / "week2_metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_name = (
        "training_log_finetuned"
        if fine_tune
        else "training_log_balanced"
        if balanced
        else "training_log_baseline"
    )
    log_path = reports_dir / f"{log_name}_{timestamp}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"Training log saved to: {log_path.resolve()}")

    # Also save root-level summary for legacy tests/plots
    pd.DataFrame(logs).to_csv("training_summary.csv", index=False)
    print("Also saved legacy training_summary.csv at project root")

    # Final model save
    final_path = Path(
        "saved_models/resnet50_finetuned.pt"
        if fine_tune
        else "saved_models/resnet50_baseline.pt"
    )
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path.resolve()}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Train PneumoDetect model.")
        parser.add_argument("--balanced", action="store_true", help="Use WeightedRandomSampler.")
        parser.add_argument("--fine_tune", action="store_true", help="Unfreeze top ResNet blocks.")
        parser.add_argument("--epochs", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument(
            "--csv_path", type=str, default="data/rsna_subset/stage_2_train_labels.csv"
        )
        parser.add_argument("--img_dir", type=str, default="data/rsna_subset/train_images")
        args = parser.parse_args()

        train_baseline(
            csv_path=args.csv_path,
            img_dir=args.img_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            balanced=args.balanced,
            fine_tune=args.fine_tune,
        )
    except Exception as e:
        print(f"Runtime error in main: {e}", file=sys.stderr)
        sys.exit(1)
