"""
Unit tests for subset_rsna.py
-----------------------------
Verifies that the subset creation script:
1. Creates expected output folders.
2. Generates a CSV file.
3. Matches image count with CSV entries.
"""

import os
import shutil
import pandas as pd
from pathlib import Path
from utility.subset_rsna import create_subset


def setup_test_data(tmp_path):
    """Create a small fake RSNA dataset for testing."""
    src_dir = tmp_path / "rsna"
    src_img_dir = src_dir / "stage_2_train_images"
    src_img_dir.mkdir(parents=True)
    src_labels = src_dir / "stage_2_train_labels.csv"

    # Create 10 dummy DICOM files
    for i in range(10):
        (src_img_dir / f"img_{i}.dcm").write_text("fake dicom data")

    # Create CSV with matching patientIds
    df = pd.DataFrame({
        "patientId": [f"img_{i}" for i in range(10)],
        "Target": [0, 1] * 5
    })
    df.to_csv(src_labels, index=False)
    return src_dir


def test_create_subset(tmp_path):
    """Test subset creation logic."""
    src_dir = setup_test_data(tmp_path)
    dst_dir = tmp_path / "subset"

    # Create subset of 5 images
    create_subset(src_dir, dst_dir, n=5)

    subset_imgs = list((dst_dir / "train_images").glob("*.dcm"))
    subset_csv = dst_dir / "train_labels_subset.csv"

    # Assertions
    assert subset_csv.exists(), "Subset CSV not created"
    assert len(subset_imgs) == 5, "Incorrect number of images copied"

    df_subset = pd.read_csv(subset_csv)
    assert len(df_subset) <= 5, "Subset CSV contains extra rows"
    assert all(df_subset["patientId"].isin([f.stem for f in subset_imgs])), \
        "CSV entries don't match copied images"