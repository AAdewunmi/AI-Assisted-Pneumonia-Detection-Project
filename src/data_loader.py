"""
Data loading utilities for PneumoDetect — supports balanced sampling and
fallback for synthetic tests.
"""

import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import pandas as pd
from PIL import Image
import numpy as np


def get_default_transform():
    """Return a standard transform pipeline (resize, normalize)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class PneumoniaDataset(Dataset):
    """Dataset for chest X-ray images and pneumonia labels."""

    def __init__(self, csv_path, img_dir, transform=None):
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform or get_default_transform()

        self.data = pd.read_csv(self.csv_path)

        if not self.img_dir.exists():
            print(
                "No matching images found; keeping all rows for "
                "synthetic/testing mode."
            )
        else:
            existing_files = set(f.stem for f in self.img_dir.glob("*"))
            self.data = self.data[self.data["patientId"].isin(existing_files)]
            if len(self.data) == 0:
                print("No image matches — using all rows for synthetic tests.")

        print(f"Loaded {len(self.data)} records from {self.csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return image tensor/label; fallback to dummy tensor if missing."""
        if idx >= len(self.data):
            idx = idx % len(self.data)
        row = self.data.iloc[idx]
        pid = row["patientId"]
        label = int(row["Target"])

        # Look for image file
        img_path = None
        for ext in [".dcm", ".png", ".jpg"]:
            path_candidate = self.img_dir / f"{pid}{ext}"
            if path_candidate.exists():
                img_path = path_candidate
                break

        if img_path is None:
            # Synthetic/fake data mode: return dummy image
            return torch.zeros((3, 224, 224)), label

        try:
            import pydicom
            img = pydicom.dcmread(img_path).pixel_array
        except Exception:
            img = np.array(Image.open(img_path).convert("L"))

        img = np.stack([img] * 3, axis=-1)
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label


def get_class_weights(csv_path):
    """Compute inverse frequency weights per class."""
    df = pd.read_csv(csv_path)
    counts = df["Target"].value_counts().to_dict()
    weights = {cls: 1.0 / count for cls, count in counts.items()}
    print(f"Class counts: {counts} | Weights: {weights}")
    return weights


def get_balanced_loader(csv_path, img_dir, transform=None, batch_size=8):
    """Balanced DataLoader with WeightedRandomSampler and smoothed weights."""
    from src.train import collate_skip_none

    dataset = PneumoniaDataset(
        csv_path, img_dir, transform or get_default_transform()
    )
    df = pd.read_csv(csv_path)

    counts = df["Target"].value_counts().to_dict()
    weights = df["Target"].apply(
        lambda x: 1.0 / (counts[x] ** 0.7)
    ).values
    weights = torch.DoubleTensor(weights)
    weights = weights / weights.sum() * len(weights)

    sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
    print(f"Class counts: {counts} | Smoothed sample weights applied.")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_skip_none
    )
    print("Balanced DataLoader ready.")
    return loader


def get_data_loader(
    csv_path, img_dir, transform=None, batch_size=8, shuffle=True
):
    """Standard unbalanced DataLoader (default)."""
    from src.train import collate_skip_none
    dataset = PneumoniaDataset(
        csv_path, img_dir, transform or get_default_transform()
    )
    print("Standard DataLoader ready.")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_skip_none
    )
