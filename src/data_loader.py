"""
Custom PyTorch Dataset and DataLoader utilities for the RSNA Pneumonia Detection subset.
Now includes WeightedRandomSampler support and a standard preprocessing transform.
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from pathlib import Path
from PIL import Image


class PneumoniaDataset(Dataset):
    """Custom dataset for loading RSNA Pneumonia Detection images and labels."""

    def __init__(self, csv_path, img_dir, transform=None):
        """
        Initialize dataset with image directory and labels CSV.

        Args:
            csv_path (str or Path): Path to CSV containing 'patientId' and 'Target' columns.
            img_dir (str or Path): Directory containing DICOM or PNG/JPG images.
            transform (callable, optional): Transformations to apply to each image.
        """
        self.csv_path = Path(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.data = pd.read_csv(self.csv_path)

        # Filter rows to only those with available images
        valid_ids = []
        for pid in self.data["patientId"]:
            if any((self.img_dir / f"{pid}{ext}").exists() for ext in [".dcm", ".png", ".jpg"]):
                valid_ids.append(pid)

        self.data = self.data[self.data["patientId"].isin(valid_ids)].reset_index(drop=True)
        print(f"Filtered to {len(self.data)} records with existing images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single sample (image, label) from the dataset.
        """
        row = self.data.iloc[idx]
        pid = row["patientId"]
        label = int(row["Target"])

        # Attempt to locate image with possible extensions
        for ext in [".dcm", ".png", ".jpg"]:
            img_path = self.img_dir / f"{pid}{ext}"
            if img_path.exists():
                break
        else:
            raise FileNotFoundError(f"No image found for ID: {pid}")

        # Load DICOM or fallback to image
        try:
            import pydicom
            if img_path.suffix == ".dcm":
                img = pydicom.dcmread(img_path).pixel_array
            else:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        except Exception:
            img = np.array(Image.open(img_path).convert("L"))

        # Convert grayscale to RGB (3 channels)
        img = np.stack([img] * 3, axis=-1)

        # Apply transform or default preprocessing
        if self.transform:
            img = self.transform(Image.fromarray(img))
        else:
            img = get_default_transform()(Image.fromarray(img))

        return img, label


def get_default_transform():
    """
    Return the default preprocessing pipeline used throughout training.

    Includes:
        - Resize to 224x224
        - Convert to tensor
        - Normalize using ImageNet mean/std
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_class_weights(csv_path):
    """
    Compute inverse frequency weights for each class.

    Args:
        csv_path (str or Path): CSV file containing 'Target' column.

    Returns:
        dict: Mapping from class label to weight.
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

    Args:
        csv_path (str or Path): Path to CSV containing labels.
        img_dir (str or Path): Path to image directory.
        transform (callable, optional): Image transformation pipeline.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: Balanced DataLoader for training.
    """
    dataset = PneumoniaDataset(csv_path, img_dir, transform)
    df = pd.read_csv(csv_path)
    weights_dict = get_class_weights(csv_path)
    sample_weights = [weights_dict[label] for label in df["Target"].values]

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
    Convenience wrapper returning a standard DataLoader.

    Args:
        csv_path (str or Path): Path to CSV containing labels.
        img_dir (str or Path): Path to image directory.
        transform (callable, optional): Transformations to apply.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle data.

    Returns:
        DataLoader: Unbalanced DataLoader for baseline training.
    """
    from src.train import collate_skip_none
    dataset = PneumoniaDataset(csv_path, img_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_skip_none)
