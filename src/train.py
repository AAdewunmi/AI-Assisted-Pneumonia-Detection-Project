"""
Training script for PneumoDetect baseline, balanced, and fine-tuned models.

Features:
- Supports unbalanced, weighted, and fine-tuned training modes.
- Differential learning rates for fine-tuning (base vs. head).
- ReduceLROnPlateau learning-rate scheduler.
- Logs loss, accuracy, and ROC-AUC per epoch.
- Saves best model and final checkpoint.
- CI-safe: generates synthetic CSV if dataset is missing.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from src.data_loader import (
    get_data_loader,
    get_balanced_loader,
    get_default_transform,
)
from src.model import build_resnet50_baseline, build_resnet50_finetuned


def collate_skip_none(batch):
    """Remove None entries (missing images) from minibatches."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels)


def ensure_dataset_available(csv_path: Path) -> Path:
    """
    Ensure dataset CSV exists.
    If missing, create a synthetic CSV for CI/testing.
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
    Saves best checkpoint and detailed training logs.
    """
    csv_path = ensure_dataset_available(Path(csv_path))
    img_dir = Path(img_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = get_default_transform()

    # Select data loader type
    if balanced:
        loader = get_balanced_loader(csv_path, img_dir, transform, batch_size=batch_size)
        print("Balanced DataLoader ready.")
    else:
        loader = get_data_loader(csv_path, img_dir, transform, batch_size=batch_size)
        print("Standard DataLoader ready.")

    # Select model type
    model = (
        build_resnet50_finetuned()
        if fine_tune
        else build_resnet50_baseline()
    ).to(device)

    # Define optimizer with differential LR if fine-tuning
    if fine_tune:
        params = [
            {"params": model.layer3.parameters(), "lr": 1e-5},
            {"params": model.layer4.parameters(), "lr": 1e-5},
            {"params": model.fc.parameters(), "lr": 1e-4},
        ]
        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0
    logs = []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)
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
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(probs.numpy())
            all_labels.extend(labels.cpu().numpy())

            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}"})

        epoch_loss = running_loss / max(1, len(loader))
        epoch_acc = correct / total if total > 0 else 0.0

        # Compute ROC-AUC safely
        try:
            epoch_auc = roc_auc_score(all_labels, all_preds)
        except Exception:
            epoch_auc = float("nan")

        logs.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "roc_auc": epoch_auc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch {epoch + 1}: Loss={epoch_loss:.4f}, "
            f"Accuracy={epoch_acc:.4f}, ROC-AUC={epoch_auc:.4f}"
        )

        if scheduler is not None:
            scheduler.step(epoch_loss)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            Path("saved_models").mkdir(parents=True, exist_ok=True)
            suffix = "_finetuned" if fine_tune else "_best"
            best_path = Path(f"saved_models/resnet50{suffix}.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (Accuracy={best_acc:.4f}) → {best_path}")

    # Save training logs
    reports_dir = Path("reports") / "week2_metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if fine_tune:
        log_name = "training_log_finetuned"
    elif balanced:
        log_name = "training_log_balanced"
    else:
        log_name = "training_log_baseline"

    log_path = reports_dir / f"{log_name}_{timestamp}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"Training log saved to: {log_path.resolve()}")

    # Backward compatibility: legacy CSVs in project root
    pd.DataFrame(logs).to_csv("training_log.csv", index=False)
    pd.DataFrame(logs).to_csv("training_summary.csv", index=False)
    print("Also saved legacy logs to project root.")

    # Save final model
    final_model_name = "resnet50_finetuned.pt" if fine_tune else "resnet50_baseline.pt"
    final_model_path = Path("saved_models") / final_model_name
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path.resolve()}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Train PneumoDetect baseline, balanced, or fine-tuned models."
        )
        parser.add_argument("--balanced", action="store_true", help="Use WeightedRandomSampler for balanced training.")
        parser.add_argument("--fine_tune", action="store_true", help="Unfreeze last 2 ResNet blocks for fine-tuning.")
        parser.add_argument("--epochs", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--csv_path", type=str, default="data/rsna_subset/stage_2_train_labels.csv")
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
