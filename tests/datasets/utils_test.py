import os
import pytest
import unittest
import yaml
import boto3
from botocore.exceptions import NoCredentialsError
from unittest.mock import patch, MagicMock
from omegaconf import OmegaConf

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir,'..','src'))
sys.path.append(os.path.join(root_dir,'..','docker'))

from czibench.datasets.utils import _download_dataset, load_dataset
from czibench.datasets.sc import SingleCellDataset

@pytest.fixture
def mock_s3_client():
    with patch("boto3.client") as mock:
        mock_s3 = mock.return_value
        yield mock_s3

@pytest.fixture
def mock_hydra():
    with patch("hydra.core.global_hydra.GlobalHydra.instance") as mock:
        yield mock

@pytest.fixture
def mock_hydra_initialize():
    with patch("hydra.initialize") as mock:
        yield mock

@pytest.fixture
def mock_hydra_compose():
    with patch("hydra.compose") as mock:
        mock.return_value = OmegaConf.create({
            "datasets": {
                "test_dataset": {
                    "_target_": "czibench.datasets.sc.SingleCellDataset",
                    "path": "/tmp/test_dataset.h5ad",
                    "organism": "HUMAN"
                }
            }
        })
        yield mock

@pytest.fixture
def mock_instantiate():
    with patch("czibench.datasets.utils.instantiate") as mock:
        mock.return_value = SingleCellDataset(path="/tmp/test_dataset.h5ad", organism="HUMAN")
        yield mock

def test_download_dataset(mock_s3_client):
    mock_s3_client.download_file.return_value = None
    _download_dataset("s3://my-bucket/my-dataset.h5ad", "/tmp/my-dataset.h5ad")

    mock_s3_client.download_file.assert_called_once_with(
        "my-bucket", "my-dataset.h5ad", "/tmp/my-dataset.h5ad"
    )

def test_download_dataset_invalid_s3(mock_s3_client):
    mock_s3_client.download_file.side_effect = NoCredentialsError

    with pytest.raises(NoCredentialsError):
        _download_dataset("s3://my-bucket/my-dataset.h5ad", "/tmp/my-dataset.h5ad")

def test_load_dataset_valid(mock_hydra, mock_hydra_initialize, mock_hydra_compose, mock_instantiate):
    with patch("os.path.exists", return_value=True):  # Mock file existence
        dataset = load_dataset("test_dataset")
        assert isinstance(dataset, SingleCellDataset)
        assert dataset.path == "/tmp/test_dataset.h5ad"

def test_load_dataset_invalid_name(mock_hydra, mock_hydra_initialize, mock_hydra_compose):
    with pytest.raises(ValueError, match="Dataset invalid_dataset not found in config"):
        load_dataset("invalid_dataset")

def test_load_dataset_missing_file(mock_hydra, mock_hydra_initialize, mock_hydra_compose):
    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="Local dataset file not found"):
            load_dataset("test_dataset")
