"""
src/data_loader.py
------------------
Custom PyTorch Dataset for the RSNA Pneumonia Detection subset.
Handles missing images gracefully and supports transforms for preprocessing.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PneumoniaDataset(Dataset):
    """
    Custom dataset for loading RSNA Pneumonia Detection images and labels.

    Args:
        csv_path (str or Path): Path to the CSV containing patientId and Target.
        img_dir (str or Path): Directory containing DICOM or PNG images.
        transform (callable, optional): Transformations applied to each image.

    Returns:
        Tuple[Tensor, int]: Transformed image tensor and corresponding label.
    """

    def __init__(self, csv_path, img_dir, transform=None):
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.data = pd.read_csv(self.csv_path)

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        print(f" Loaded {len(self.data)} records from {self.csv_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pid = row["patientId"]
        label = int(row["Target"])

        img_path_dcm = self.img_dir / f"{pid}.dcm"
        img_path_png = self.img_dir / f"{pid}.png"

        # Handle missing files gracefully
        if not img_path_dcm.exists() and not img_path_png.exists():
            print(f"Missing image for {pid}")
            return None

        try:
            import pydicom
            if img_path_dcm.exists():
                img = pydicom.dcmread(img_path_dcm).pixel_array
            else:
                img = cv2.imread(str(img_path_png), cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print(f"Failed to load {pid}: {e}")
            return None

        # Convert grayscale → 3 channels
        img = np.stack([img] * 3, axis=-1)

        if self.transform:
            img = self.transform(img)

        return img, label


def get_data_loader(csv_path, img_dir, transform=None, batch_size=8, shuffle=True):
    """
    Convenience function for tests — returns a DataLoader for PneumoniaDataset.
    Skips missing images gracefully.
    """
    from src.train import collate_skip_none  # lazy import to avoid circular dependency
    dataset = PneumoniaDataset(csv_path, img_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_skip_none)
