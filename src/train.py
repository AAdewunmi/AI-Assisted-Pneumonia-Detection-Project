"""
src/train.py
------------
Training script for the baseline ResNet-50 pneumonia classifier.
Performs transfer learning with frozen backbone and fine-tuned classifier head.
Logs training metrics (loss, accuracy) using tqdm.
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
from src.model import build_resnet50_baseline
from src.data_loader import PneumoniaDataset


def train_baseline(csv_path: str, img_dir: str, epochs: int = 5, batch_size: int = 8, lr: float = 1e-3):
    """
    Trains the ResNet-50 baseline model on the pneumonia dataset.

    Args:
        csv_path (str): Path to labels CSV.
        img_dir (str): Directory with training images.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        lr (float): Learning rate.
    """
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Preprocessing transforms (same as data_loader)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Dataset & DataLoader
    dataset = PneumoniaDataset(csv_path, img_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = build_resnet50_baseline(num_classes=2, freeze_backbone=True).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    # Training loop
    model.train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")

        for imgs, labels in pbar:
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

        print(f"Epoch {epoch+1}: Loss={running_loss/len(loader):.4f}, Accuracy={correct/total:.4f}")

    # Save trained weights
    save_dir = Path("saved_models")
    save_dir.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), save_dir / "resnet50_baseline.pt")
    print("Model saved to:", save_dir / "resnet50_baseline.pt")


if __name__ == "__main__":
    csv_path = "data/rsna_subset/train_labels_subset.csv"
    img_dir = "data/rsna_subset/train_images"
    train_baseline(csv_path, img_dir, epochs=3, batch_size=8)
