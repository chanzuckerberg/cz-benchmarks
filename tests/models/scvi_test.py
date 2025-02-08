import os
import sys
import pathlib
import pytest
import pandas as pd
import numpy as np
import anndata as ad
from omegaconf import OmegaConf
from unittest import mock
import boto3
import botocore
root_dir = pathlib.Path(__file__).parent.parent.parent
sys.path.append(os.path.join(root_dir, 'src'))
sys.path.append(os.path.join(root_dir, 'docker'))

from src.czibench.datasets.sc import SingleCellDataset
from docker.scvi.model import SCVI
from src.czibench.datasets.utils import load_dataset
from src.czibench.datasets.types import Organism

from src.czibench.constants import (
    INPUT_DATA_PATH_DOCKER,
    RAW_INPUT_DIR_PATH_DOCKER, 
    OUTPUT_DATA_PATH_DOCKER,
    ARTIFACTS_PATH_DOCKER,
    DATASETS_CACHE_PATH, 
    MODEL_WEIGHTS_PATH_DOCKER,
    MODEL_WEIGHTS_CACHE_PATH,
)

TEMPORARY_PATH = pathlib.Path("/tmp/cz_benchmarks_test")
S3_DATASET_PATH = "s3://generate-cross-species/datasets/test/example-small.h5ad"
S3_MODEL_WEIGHTS_BASE_PATH = "s3://cellxgene-contrib-public/models/scvi/2024-07-01"

def get_model_config():
    """
    Load the model configuration from the specified YAML file.

    Returns:
        OmegaConf: The loaded configuration object.
    """
    config_path = "docker/scvi/config.yaml"
    return OmegaConf.load(config_path)


def setup_directories_for_test():
    """
    Create the directories required for the above functions to work correctly.
    Skip if the directory already exists.
    """
    directories = [
        MODEL_WEIGHTS_PATH_DOCKER,
        os.path.join(MODEL_WEIGHTS_PATH_DOCKER, "homo_sapiens"),
        os.path.join(MODEL_WEIGHTS_PATH_DOCKER, "mus_musculus"),
        DATASETS_CACHE_PATH,
        ARTIFACTS_PATH_DOCKER,
        RAW_INPUT_DIR_PATH_DOCKER,
        './app/output/',
        './app/input/',
    ]

    for directory in directories:
        if not os.path.exists(directory):
            # new_directory = os.path.join(TEMPORARY_PATH, directory)
            os.makedirs(directory)

def download_file_from_s3(s3_path: str, local_path: str):
    """
    Download a file from S3 to a local path.

    Args:
        s3_path (str): The S3 path (e.g., 's3://bucket/key').
        local_path (str): The local file path to save the downloaded file.
    """
    s3 = boto3.client('s3')
    bucket, key = parse_s3_path(s3_path)
    try:
        s3.download_file(bucket, key, local_path)
    except botocore.exceptions.ClientError as e:
        raise FileNotFoundError(f"Could not download {s3_path}: {e}")

def parse_s3_path(s3_path: str):
    """
    Parse an S3 path into bucket and key.

    Args:
        s3_path (str): The S3 path (e.g., 's3://bucket/key').

    Returns:
        tuple: (bucket, key)
    """
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with 's3://'")
    parts = s3_path[5:].split('/', 1)
    if len(parts) != 2:
        raise ValueError("S3 path must be in the format 's3://bucket/key'")
    return parts[0], parts[1]

def download_h5ad_files():
    """
    Fixture to download the necessary h5ad files from S3 if they don't exist locally.
    This includes the dataset and model weights.

    Args:
        tmp_path_factory: A pytest fixture that provides temporary directories.

    Returns:
        str: The local path where the dataset file is stored.
    """
    local_dataset_path = pathlib.Path(RAW_INPUT_DIR_PATH_DOCKER) / "example-small.h5ad"
    
    if not local_dataset_path.exists():
        download_file_from_s3(S3_DATASET_PATH, str(local_dataset_path))
    
    return str(local_dataset_path)


def download_model_weights():
    """
    Download model weights if not already present.
    """
    setup_directories_for_test()

    config =  get_model_config()
    s3_model_weights_path = config.homo_sapiens.model_weights
    model_weights_path = os.path.join(MODEL_WEIGHTS_PATH_DOCKER, str(Organism.HUMAN), "model.pt")
    if not os.path.exists(model_weights_path):
        download_file_from_s3(s3_model_weights_path, model_weights_path)
    else:
        print(f"Model weights already present at {model_weights_path}")


def test_get_model_config():
    """
    Test the get_model_config function to ensure it loads the configuration correctly.
    """
    config = get_model_config()
    assert config is not None, "Config should not be None"
    assert 'homo_sapiens' in config, "Config should contain 'homo_sapiens' key"
    assert 'model_weights' in config.homo_sapiens, "Config should contain 'model_weights' key under 'homo_sapiens'"


def test_download_model_weights(tmp_path):
    """
    Test the download_model_weights function to ensure it downloads the model weights correctly.
    """

    download_model_weights()
    model_weights_dir = os.path.join(MODEL_WEIGHTS_PATH_DOCKER, str(Organism.HUMAN))
    model_pt_path = os.path.join(model_weights_dir, "model.pt")

    assert os.path.exists(model_pt_path), f"Model weights should be downloaded to {model_pt_path}"

def test_download_h5ad_files(tmp_path_factory):
    """
    Test the download_h5ad_files function to ensure it downloads the h5ad files correctly.
    """
    # Mock the directory paths to use the temporary path

    local_dataset_path = download_h5ad_files()

    assert pathlib.Path(local_dataset_path).exists(), f"h5ad file should be downloaded to {local_dataset_path}"


def setup_hvg_files():
    """
    Use the real HVG CSV file for homo_sapiens.
    """
    # Assuming HVG file is already present in 'docker/scvi/hvg_names_homo_sapiens.csv.gz'
    hvg_human = pathlib.Path("docker/scvi/hvg_names_homo_sapiens.csv.gz")          
    if not hvg_human.exists():
        raise FileNotFoundError(f"HVG file not found at {hvg_human}")


def test_setup_hvg_files(tmp_path):
    """
    Test the setup_hvg_files function to ensure it sets up the HVG files correctly.
    """
    assert os.path.exists('docker/scvi/hvg_names_homo_sapiens.csv.gz'), f"HVG file exists"


def test_dataset_validation():
    """
    Test that the SingleCellDataset validation works correctly.
    """
    local_dataset_path = pathlib.Path(RAW_INPUT_DIR_PATH_DOCKER) / "example-small.h5ad"
    assert local_dataset_path.exists(), f"Downloaded dataset exists in {local_dataset_path}"
    dataset = SingleCellDataset(local_dataset_path, Organism.HUMAN)
    dataset.load_data()
    dataset.validate()
    dataset.serialize(INPUT_DATA_PATH_DOCKER)

    # Assertions
    assert dataset.adata is not None, "AnnData object should be loaded"
    assert dataset.output_embedding is None, "Output embedding should be None before running the model"
    assert os.path.exists(INPUT_DATA_PATH_DOCKER), "Serialized SingleCellDataset file exist"

def test_scvi_model_run():
    """
    Test running the SCVI model and saving the output embedding.
    """
    assert os.path.exists(INPUT_DATA_PATH_DOCKER), "Serialized output dataset should exist."

    scvi_model = SCVI()
    scvi_model.run()

    loaded_dataset = scvi_model.data.output_embedding
    print("SCVI model output embedding", loaded_dataset)
    assert loaded_dataset is not None, "Deserialized output embedding should be generated"
    assert isinstance(loaded_dataset, np.ndarray), "Deserialized output embedding should be a NumPy array"
