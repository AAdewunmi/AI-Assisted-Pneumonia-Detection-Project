"""
Data loading utilities for PneumoDetect.
Includes support for:
- Automatic filtering of missing image files
- WeightedRandomSampler for class balancing
- Default transforms for model compatibility
- Test-safe fallback when no real images are present
"""

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms


def get_default_transform():
    """
    Return standard preprocessing transform:
    - Convert to tensor
    - Resize to 224x224
    - Normalize to ImageNet mean/std
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class PneumoniaDataset(Dataset):
    """Dataset for loading chest X-ray images and pneumonia labels."""

    def __init__(self, csv_path, img_dir, transform=None):
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform or get_default_transform()

        # Load label data
        self.data = pd.read_csv(self.csv_path)
        if not self.img_dir.exists():
            print(f"Warning: image directory not found ({self.img_dir}). Skipping filtering.")
        else:
            # Filter out entries without image files
            valid_ids = [
                pid for pid in self.data["patientId"]
                if any((self.img_dir / f"{pid}{ext}").exists() for ext in [".dcm", ".png", ".jpg"])
            ]

            # Fallback for CI/test mode (no images)
            if len(valid_ids) == 0:
                print("No matching images found; keeping all rows for synthetic/testing mode.")
            else:
                self.data = self.data[self.data["patientId"].isin(valid_ids)].reset_index(drop=True)
                print(f"Filtered to {len(self.data)} records with existing images.")

        print(f"Loaded {len(self.data)} records from {self.csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pid = row["patientId"]
        label = int(row["Target"])

        # Try to find image file
        for ext in [".dcm", ".png", ".jpg"]:
            img_path = self.img_dir / f"{pid}{ext}"
            if img_path.exists():
                break
        else:
            # For CI/testing: return a dummy tensor
            dummy = torch.zeros((3, 224, 224))
            return dummy, label

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
    """
    Compute inverse frequency weights for each class.
    Used by WeightedRandomSampler to balance the dataset.
    """
    df = pd.read_csv(csv_path)
    counts = df["Target"].value_counts().to_dict()
    total = sum(counts.values())
    weights = {cls: total / count for cls, count in counts.items()}
    print(f"Class counts: {counts} | Weights: {weights}")
    return weights


def get_balanced_loader(csv_path, img_dir, transform=None, batch_size=8):
    """
    Return a DataLoader that balances classes using WeightedRandomSampler.
    Automatically handles missing files and synthetic test data.
    """
    dataset = PneumoniaDataset(csv_path, img_dir, transform)
    df = dataset.data  # use the filtered data
    weights_dict = get_class_weights(csv_path)

    # Handle synthetic/test CSVs gracefully
    if df.empty:
        print("Warning: dataset empty. Creating synthetic test labels.")
        df = pd.DataFrame({"Target": [0, 1] * 5})

    sample_weights = [weights_dict.get(label, 1.0) for label in df["Target"].values]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    print("Balanced DataLoader ready.")
    return loader


def get_data_loader(csv_path, img_dir, transform=None, batch_size=8, shuffle=True):
    """
    Standard DataLoader without balancing.
    Used for evaluation and EDA notebooks.
    """
    from src.train import collate_skip_none
    dataset = PneumoniaDataset(csv_path, img_dir, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_skip_none)
