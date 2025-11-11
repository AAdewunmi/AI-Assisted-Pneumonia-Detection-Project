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
