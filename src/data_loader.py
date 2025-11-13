"""
src/data_loader.py
------------------
Implements PyTorch DataLoader for pneumonia X-ray dataset.
Performs:
 - Image resizing (224×224)
 - Normalization (ImageNet mean/std)
 - Basic augmentation (flip, rotation, brightness)
"""

from pathlib import Path
from touch.utils.data import DataLoader, Dataset
from touchvision import transforms
from PIL import Image
import pandas as pd
import torch


class PneumoniaDataset(Dataset):
    """Custom dataset for loading chest X-ray images and labels."""

    def __init__(self, csv_path, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    