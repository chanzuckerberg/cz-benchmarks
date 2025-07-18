#!/usr/bin/env python3
"""
Script to download and process embedding files from S3.
Downloads dill files from S3 and extracts numpy embeddings.
"""

import subprocess
import sys
from pathlib import Path
import dill
import numpy as np

# S3 locations and corresponding output files
EMBEDDING_FILES = [
    {
        "s3_url": "s3://cz-benchmarks-results-dev/v0.1.0/processed-datasets/tsv2_bone_marrow_UCE_model_variant-4l.dill",
        "dill_filename": "tsv2_bone_marrow_UCE_model_variant-4l.dill",
        "npy_filename": "tsv2_bone_marrow_UCE_model_variant-4l-embedding.npy",
    },
    {
        "s3_url": "s3://cz-benchmarks-results-dev/v0.1.0/processed-datasets/human_spermatogenesis_UCE_model_variant-4l.dill",
        "dill_filename": "human_spermatogenesis_UCE_model_variant-4l.dill",
        "npy_filename": "human_spermatogenesis_UCE_model_variant-4l.npy",
    },
    {
        "s3_url": "s3://cz-benchmarks-results-dev/v0.1.0/processed-datasets/mouse_spermatogenesis_UCE_model_variant-4l.dill",
        "dill_filename": "mouse_spermatogenesis_UCE_model_variant-4l.dill",
        "npy_filename": "mouse_spermatogenesis_UCE_model_variant-4l.npy",
    },
    {
        "s3_url": "s3://cz-benchmarks-results-dev/v0.1.0/processed-datasets/rhesus_macaque_spermatogenesis_UCE_model_variant-4l.dill",
        "dill_filename": "rhesus_macaque_spermatogenesis_UCE_model_variant-4l.dill",
        "npy_filename": "rhesus_macaque_spermatogenesis_UCE_model_variant-4l.npy",
    },
]


def download_from_s3(s3_url, local_path):
    """Download file from S3 using AWS CLI."""
    try:
        subprocess.run(
            ["aws", "s3", "cp", s3_url, str(local_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✓ Downloaded {s3_url} to {local_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to download {s3_url}: {e.stderr}")
        return False


def extract_embedding(dill_path, npy_path):
    """Extract numpy embedding from dill file and save to disk."""
    try:
        with open(dill_path, "rb") as file:
            embedding = dill.load(file)
            np.save(npy_path, embedding.outputs["UCE"]["EMBEDDING"])
        print(f"✓ Extracted embedding from {dill_path} to {npy_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to extract embedding from {dill_path}: {e}")
        return False


def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    embeddings_dir = script_dir / "embeddings"

    # Create embeddings directory if it doesn't exist
    embeddings_dir.mkdir(exist_ok=True)

    print(f"Processing {len(EMBEDDING_FILES)} embedding files...")

    success_count = 0
    for file_info in EMBEDDING_FILES:
        print(f"\n--- Processing {file_info['dill_filename']} ---")

        dill_path = embeddings_dir / file_info["dill_filename"]
        npy_path = embeddings_dir / file_info["npy_filename"]

        # Skip if npy file already exists
        if npy_path.exists():
            print(f"✓ {npy_path.name} already exists, skipping...")
            success_count += 1
            continue

        # Download dill file if it doesn't exist
        if not dill_path.exists():
            if not download_from_s3(file_info["s3_url"], dill_path):
                continue
        else:
            print(f"✓ {dill_path.name} already exists, skipping download")

        # Extract embedding
        if extract_embedding(dill_path, npy_path):
            success_count += 1

    print("\n=== Summary ===")
    print(f"Successfully processed {success_count}/{len(EMBEDDING_FILES)} files")

    if success_count == len(EMBEDDING_FILES):
        print("All embeddings processed successfully!")
        return 0
    else:
        print("Some embeddings failed to process.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
