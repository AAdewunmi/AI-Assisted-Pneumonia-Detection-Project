"""
src/data_loader.py
------------------
Custom PyTorch Dataset and DataLoader utilities for the RSNA Pneumonia Detection subset.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class PneumoniaDataset(Dataset):
    """Custom dataset for loading RSNA Pneumonia Detection images and labels."""

    def __init__(self, csv_path, img_dir, transform=None):
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.data = pd.read_csv(self.csv_path)

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        print(f"Loaded {len(self.data)} records from {self.csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pid = row["patientId"]
        label = int(row["Target"])

        img_path_dcm = self.img_dir / f"{pid}.dcm"
        img_path_png = self.img_dir / f"{pid}.png"

        if not img_path_dcm.exists() and not img_path_png.exists():
            return None

        try:
            import pydicom
            if img_path_dcm.exists():
                img = pydicom.dcmread(img_path_dcm).pixel_array
            else:
                img = cv2.imread(str(img_path_png), cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None

        img = np.stack([img] * 3, axis=-1)  # make 3 channels

        if self.transform:
            img = self.transform(img)
        else:
            # default normalization to match preprocessing test
            tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            img = tf(img)

        return img, label


def get_data_loader(csv_path, img_dir, transform=None, batch_size=8, shuffle=True):
    """Convenience wrapper for tests."""
    from src.train import collate_skip_none
    dataset = PneumoniaDataset(csv_path, img_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_skip_none)
