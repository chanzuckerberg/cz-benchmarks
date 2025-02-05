import pytest
import pathlib
import boto3
import anndata as ad
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir,'..','src'))
sys.path.append(os.path.join(root_dir,'..','docker'))

from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism
from scvi.model import SCVI



TEST_H5AD_PATH = "tests/datasets/Homo_sapiens_ERP132584.h5ad"
TEST_MODEL_WEIGHTS_PATH = "tests/models/scvi_model.pt"

@pytest.fixture
def test_dataset():
    """Fixture to load the test dataset from local file."""
    dataset = SingleCellDataset(path=TEST_H5AD_PATH, organism=Organism.HUMAN)
    dataset.load_data()  
    return dataset

@pytest.fixture
def scvi_model(test_dataset):
    """Fixture to initialize the SCVI model with local weights."""
    scvi = SCVI()
    scvi.data = test_dataset
    scvi.model_weights_dir = pathlib.Path("tests/models")  
    return scvi

def test_scvi_class_attributes():
    """Test SCVI class attributes."""
    assert SCVI.available_organisms == [Organism.HUMAN, Organism.MOUSE]
    assert SCVI.required_obs_keys == ["dataset_id", "assay", "suspension_type", "donor_id"]
    assert SCVI.required_var_keys == []

def test_validate_model_requirements(test_dataset):
    """Test model requirement validation for SCVI."""
    assert SCVI._validate_model_requirements(test_dataset) is True
    test_dataset.adata.obs.drop(columns=["dataset_id"], inplace=True)
    with pytest.raises(ValueError, match="Missing required obs keys"):
        SCVI._validate_model_requirements(test_dataset)

def test_validate_dataset(scvi_model):
    """Test dataset validation within SCVI model."""
    scvi_model._validate_dataset(scvi_model.data)
    scvi_model.data.organism = "Unknown"
    with pytest.raises(ValueError, match="Dataset organism .* is not supported"):
        scvi_model._validate_dataset(scvi_model.data)

def test_load_local_model_weights(scvi_model):
    """Test if SCVI model correctly uses local model weights."""
    model_weights_file = scvi_model.model_weights_dir / "scvi_model.pt"
    assert model_weights_file.exists(), f"Model weights not found at {model_weights_file}"

def test_run_model(scvi_model):
    """Run the SCVI model using the actual dataset and weights."""
    scvi_model.run_model()  
    assert scvi_model.data.output_embedding is not None, "No output embedding generated"
    assert scvi_model.data.output_embedding.shape[0] > 0, "Output embedding is empty"