"""
subset_rsna.py
----------------
Create a manageable (~2 GB) subset of the RSNA Pneumonia Detection dataset.
It copies ~2000 random images and their matching labels into a new folder.

Usage:
    conda activate pneumodetect
    python scripts/subset_rsna.py --src data/rsna --dst data/rsna_subset \\
        --n 2000

This will:

    Copy ~2000 random .dcm files into data/rsna_subset/train_images/
    Create data/rsna_subset/train_labels_subset.csv with matching labels.
    Print summary info (counts, estimated size)
"""

import argparse
import random
import shutil
import pandas as pd
from pathlib import Path


def create_subset(src_dir: Path, dst_dir: Path, n: int = 2000):
    """Create a random subset of RSNA images and matching labels."""
    src_images = src_dir / "stage_2_train_images"
    src_labels = src_dir / "stage_2_train_labels.csv"
    dst_images = dst_dir / "train_images"
    dst_labels = dst_dir / "train_labels_subset.csv"

    dst_images.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = pd.read_csv(src_labels)

    # Get all image filenames
    all_images = list(src_images.glob("*.dcm"))
    total_images = len(all_images)

    n = min(n, total_images)
    print(f"Total images available: {total_images}")
    print(f"Sampling {n} images (~{n * 1.0:.2f} GB estimated)")

    # Random sample and copy
    sampled_files = random.sample(all_images, n)
    for f in sampled_files:
        shutil.copy(f, dst_images / f.name)

    # Filter CSV for selected files (match by ID without extension)
    sampled_ids = [f.stem for f in sampled_files]
    df_subset = df[df["patientId"].isin(sampled_ids)]
    df_subset.to_csv(dst_labels, index=False)

    print(f"Subset complete: {len(df_subset)} records written to {dst_labels}")
    print(f"Images saved in: {dst_images.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Create RSNA subset")
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/rsna"),
        help="Source RSNA dataset folder",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/rsna_subset"),
        help="Destination subset folder",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Number of images to sample (~2 GB)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_subset(args.src, args.dst, args.n)
