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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
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

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pid = row['patientId']
        label = int(row['Target'])
        img_path_dcm = self.img_dir / f"{pid}.dcm"
        img_path_png = self.img_dir / f"{pid}.png"

        # Gracefully skip missing files
        if not img_path_dcm.exists() and not img_path_png.exists():
            # Skip bad samples
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

        img = np.stack([img] * 3, axis=-1)  # make 3 channels
        if self.transform:
            img = self.transform(img)
        return img, label



def get_data_loader(csv_path, img_dir, batch_size=8, num_workers=0):
    """
    Returns a DataLoader with preprocessing and augmentation applied.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet normalization
            std=[0.229, 0.224, 0.225]
        ),
    ])

    dataset = PneumoniaDataset(csv_path, img_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
