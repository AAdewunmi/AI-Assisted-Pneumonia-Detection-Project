"""
src/train.py
-------------
Unified training script for PneumoDetect models (baseline & fine-tuned).

Key features:
- Supports baseline or fine-tuned ResNet-50 variants automatically.
- Handles balanced vs unbalanced DataLoaders.
- Includes ReduceLROnPlateau learning-rate scheduler.
- Automatically detects latest checkpoint type (baseline vs finetuned).
- Saves both timestamped and legacy CSV logs at the project root.

Run:
    python -m src.train --epochs 3 --batch_size 8 --lr 1e-3
    python -m src.train --balanced
    python -m src.train --resume saved_models/resnet50_finetuned.pt
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from src.data_loader import (
    get_data_loader,
    get_balanced_loader,
    get_default_transform,
)
from src.model import build_resnet50_baseline, build_resnet50_finetuned


def collate_skip_none(batch):
    """Remove None samples (missing images) from minibatches."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    imgs, labels = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels)


def ensure_dataset_available(csv_path: Path) -> Path:
    """Ensure dataset CSV exists or create a small synthetic version for CI/testing."""
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


def detect_model_from_checkpoint(ckpt_path: Path):
    """
    Detect whether to load baseline or fine-tuned model based on filename.
    Returns the appropriate model constructor.
    """
    name = ckpt_path.name.lower()
    if "finetune" in name or "fine" in name:
        print("Detected fine-tuned checkpoint.")
        return build_resnet50_finetuned
    else:
        print("Detected baseline checkpoint.")
        return build_resnet50_baseline


def train_baseline(
    csv_path,
    img_dir,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 1e-3,
    balanced: bool = False,
    resume: str = None,
):
    """
    Train a baseline or fine-tuned ResNet-50 model.

    Args:
        csv_path: Path to labels CSV.
        img_dir: Directory with training images.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate for optimizer.
        balanced (bool): Use weighted sampler.
        resume (str): Optional path to checkpoint for resuming training.
    """
    csv_path = ensure_dataset_available(Path(csv_path))
    img_dir = Path(img_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = get_default_transform()
    loader = (
        get_balanced_loader(csv_path, img_dir, transform, batch_size=batch_size)
        if balanced
        else get_data_loader(csv_path, img_dir, transform, batch_size=batch_size)
    )

    # --- Load or initialize model ---
    if resume:
        ckpt_path = Path(resume)
        model_builder = detect_model_from_checkpoint(ckpt_path)
        model = model_builder().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Resumed training from checkpoint: {ckpt_path.name}")
    else:
        model = build_resnet50_baseline().to(device)

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
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            acc = correct / total if total > 0 else 0.0
            pbar.set_postfix(
                {"loss": f"{loss.item():.3f}", "acc": f"{acc:.3f}", "lr": f"{lr:.6f}"}
            )

        epoch_loss = running_loss / max(1, len(loader))
        epoch_acc = correct / total if total > 0 else 0.0
        scheduler.step(epoch_loss)
        lr_current = optimizer.param_groups[0]["lr"]

        print(
            "Epoch {epoch}: Loss={loss:.4f}, Accuracy={acc:.4f}, LR={lr:.6f}".format(
                epoch=epoch + 1, loss=epoch_loss, acc=epoch_acc, lr=lr_current
            )
        )

        logs.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": epoch_acc,
                "lr": lr_current,
            }
        )

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            Path("saved_models").mkdir(parents=True, exist_ok=True)
            best_path = Path("saved_models/resnet50_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved (Accuracy={best_acc:.4f}) → {best_path}")

    # --- Save logs ---
    reports_dir = Path("reports") / "week2_metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = "training_log_balanced" if balanced else "training_log_baseline"
    log_path = reports_dir / f"{log_name}_{timestamp}.csv"
    pd.DataFrame(logs).to_csv(log_path, index=False)
    print(f"Training log saved to: {log_path.resolve()}")

    # Legacy compatibility
    legacy_summary = Path("training_summary.csv")
    pd.DataFrame(logs).to_csv(legacy_summary, index=False)
    print("Also saved legacy training_summary.csv at project root")

    # --- Save final model ---
    final_model_path = Path("saved_models/resnet50_baseline.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to: {final_model_path.resolve()}")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Train PneumoDetect ResNet-50 model.")
        parser.add_argument("--balanced", action="store_true")
        parser.add_argument("--epochs", type=int, default=3)
        parser.add_argument("--batch_size", type=int, default=8)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--resume", type=str, default=None)
        parser.add_argument(
            "--csv_path", type=str, default="data/rsna_subset/stage_2_train_labels.csv"
        )
        parser.add_argument(
            "--img_dir", type=str, default="data/rsna_subset/train_images"
        )
        args = parser.parse_args()

        train_baseline(
            csv_path=args.csv_path,
            img_dir=args.img_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            balanced=args.balanced,
            resume=args.resume,
        )
    except Exception as e:
        print(f"Runtime error in main: {e}", file=sys.stderr)
        sys.exit(1)
