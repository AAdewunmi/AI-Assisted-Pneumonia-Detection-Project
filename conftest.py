import sys
from pathlib import Path
"""
Pytest configuration for PneumoDetect project.
Provides reusable fixtures for synthetic datasets and temp image directories.
Ensures tests run cleanly without large RSNA data dependencies.
"""

import pytest
import pandas as pd
import numpy as np
from PIL import Image

# Add project root to sys.path so tests can import project modules
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def fake_csv(tmp_path_factory):
    """
    Create a small synthetic CSV similar to the RSNA pneumonia dataset.
    Returns path to the CSV.
    """
    tmp_dir = tmp_path_factory.mktemp("fake_data")
    csv_path = tmp_dir / "labels.csv"

    df = pd.DataFrame({
        "patientId": [f"fake_{i}" for i in range(10)],
        "Target": np.random.randint(0, 2, 10)
    })
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture(scope="session")
def fake_img_dir(tmp_path_factory):
    """
    Create a temporary directory with fake 224x224 grayscale PNGs.
    Simulates chest X-rays for testing.
    """
    img_dir = tmp_path_factory.mktemp("fake_images")

    for i in range(10):
        img = Image.fromarray(
            np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        )
        img.save(img_dir / f"fake_{i}.png")

    return img_dir


@pytest.fixture(scope="session")
def fake_dataset(fake_csv, fake_img_dir):
    """
    Convenience fixture returning both CSV and image directory paths.
    """
    return str(fake_csv), str(fake_img_dir)
