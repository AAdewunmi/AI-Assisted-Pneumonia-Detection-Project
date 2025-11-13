"""
src/train.py
------------
Trains the baseline ResNet-50 model on the RSNA Pneumonia Detection subset.
Includes:
 - graceful skipping of missing images
 - CSV logging of metrics
 - automatic saving of best-performing model checkpoint
"""

import csv
from datetime import datetime
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from src.model import build_resnet50_baseline
from src.data_loader import PneumoniaDataset


def collate_skip_none(batch):
    """Skip None entries returned by the Dataset."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def train_baseline(csv_path: str, img_dir: str, epochs: int = 3, batch_size: int = 8, lr: float = 1e-3):
    """Train a ResNet-50 pneumonia classifier and log metrics."""
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Image preprocessing pipeline
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset + DataLoader
    dataset = PneumoniaDataset(csv_path, img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_skip_none)

    # Model setup
    model = build_resnet50_baseline(num_classes=2, freeze_backbone=True).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    # CSV log setup
    log_path = Path("training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "epoch", "loss", "accuracy"])

    save_dir = Path("saved_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0  # track best accuracy

    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

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

            pbar.set_postfix(loss=loss.item(), acc=correct / total)

        epoch_loss = running_loss / len(loader)
        epoch_acc = correct / total
        print(f"📊 Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

        # Save metrics to log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), epoch + 1, epoch_loss, epoch_acc])

        # --- Save best model checkpoint ---
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_path = save_dir / "resnet50_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"💾 New best model saved (Accuracy={best_acc:.4f}) → {best_path}")

    # Save final model
    final_path = save_dir / "resnet50_baseline.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to: {final_path}")
    print(f"Training log saved to: {log_path}")

    # Optional: quick plot
    try:
        df = pd.read_csv(log_path)
        plt.figure(figsize=(6, 4))
        plt.plot(df["epoch"], df["accuracy"], label="Accuracy")
        plt.plot(df["epoch"], df["loss"], label="Loss")
        plt.title("Training Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot training metrics: {e}")


if __name__ == "__main__":
    csv_path = "data/rsna_subset/stage_2_train_labels.csv"
    img_dir = "data/rsna_subset/train_images"
    train_baseline(csv_path, img_dir)
