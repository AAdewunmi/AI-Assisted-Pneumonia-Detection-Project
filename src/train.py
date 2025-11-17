"""
Training script for PneumoDetect baseline and balanced models.

Features:
- Supports unbalanced and weighted sampling modes
- Automatically saves best model checkpoint (highest accuracy)
- Logs loss/accuracy per epoch to training_log.csv
- CI-safe: falls back to synthetic CSV when dataset unavailable
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm import tqdm
from pathlib import Path
from src.data_loader import get_data_loader, get_balanced_loader, get_default_transform


def collate_skip_none(batch):
    """Skip None samples (missing images)."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels)


def build_resnet50_baseline(num_classes=2):
    """Return a pretrained ResNet-50 with custom classification head."""
    model = models.resnet50(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def ensure_dataset_available(csv_path: Path):
    """
    Ensure the dataset CSV exists.
    If missing (e.g., in CI), create a small synthetic CSV for testing.
    """
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found. Creating synthetic test CSV for CI.")
        tmp_dir = Path("data")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = tmp_dir / "tmp_labels.csv"

        df = pd.DataFrame({
            "patientId": [f"fake_{i}" for i in range(10)],
            "Target": [0, 1] * 5
        })
        df.to_csv(tmp_path, index=False)
        csv_path = tmp_path
    return csv_path


def train_baseline(csv_path, img_dir, epochs=3, batch_size=8, lr=1e-3, balanced=False):
    """
    Train baseline or balanced ResNet-50 model.
    Saves best checkpoint and training log.
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

    model = build_resnet50_baseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

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
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}"})

        epoch_loss = running_loss / len(loader)
        epoch_acc = correct / total if total > 0 else 0.0

        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

        logs.append({"epoch": epoch + 1, "loss": epoch_loss, "accuracy": epoch_acc})

        # Save best model checkpoint
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            Path("saved_models").mkdir(parents=True, exist_ok=True)
            best_model_path = Path("saved_models/resnet50_best.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved (Accuracy={best_acc:.4f}) → {best_model_path}")

    # Save final model and log
    final_path = Path("saved_models/resnet50_baseline.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")

    pd.DataFrame(logs).to_csv("training_log.csv", index=False)
    print("Training log saved to: training_log.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PneumoDetect baseline model.")
    parser.add_argument("--balanced", action="store_true", help="Use WeightedRandomSampler for balanced training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
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
    )
