import pytest
import os
import anndata as ad
import pandas as pd  

import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir,'..','src'))
sys.path.append(os.path.join(root_dir,'..','docker'))

from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism


TEST_H5AD_PATH = "./tests/datasets/Homo_sapiens_ERP132584.h5ad"
SERIALIZED_PATH = "./tests/datasets/test_dataset.dill"

@pytest.fixture
def sc_dataset():
    """Fixture to create a SingleCellDataset instance."""
    dataset = SingleCellDataset(path=TEST_H5AD_PATH, organism=Organism.HUMAN)
    dataset.load_data()
    yield dataset
    dataset.unload_data()


def test_load_data(sc_dataset):
    """Test that data is loaded correctly."""
    assert sc_dataset.adata is not None, "AnnData object should be loaded"
    assert isinstance(sc_dataset.sample_metadata, pd.DataFrame), "Metadata should be a DataFrame"
    assert not sc_dataset.sample_metadata.empty, "Metadata should not be empty"


def test_unload_data(sc_dataset):
    """Test that data is unloaded correctly."""
    sc_dataset.unload_data()
    assert sc_dataset.adata is None, "AnnData object should be None after unload"


def test_validate_success(sc_dataset):
    """Test that validate succeeds for a correctly formatted dataset."""
    sc_dataset.validate()  


def test_validate_invalid_organism():
    """Test validation failure due to incorrect organism."""
    dataset = SingleCellDataset(path=TEST_H5AD_PATH, organism="INVALID")
    with pytest.raises(ValueError, match="Dataset does not contain anndata object"):
        dataset.validate()


@pytest.fixture
def test_validate_invalid_gene_prefix(monkeypatch):
    """Test validation failure when gene names do not match organism prefix."""
    dataset = SingleCellDataset(path=TEST_H5AD_PATH, organism=Organism.HUMAN)
    dataset.load_data()
    monkeypatch.setattr(dataset.adata, "var_names", new_callable=lambda: pd.Index(["XYZ0001", "XYZ0002"]))  
    with pytest.raises(ValueError, match="Dataset does not contain valid gene names"):
        dataset.validate()


def test_serialize_and_deserialize(sc_dataset):
    """Test serialization and deserialization."""
    sc_dataset.serialize(SERIALIZED_PATH)
    assert os.path.exists(SERIALIZED_PATH), "Serialized file should exist"
    
    loaded_dataset = SingleCellDataset.deserialize(SERIALIZED_PATH)
    assert isinstance(loaded_dataset, SingleCellDataset), "Deserialized object should be a SingleCellDataset"
    assert loaded_dataset.path == TEST_H5AD_PATH, "Path should be preserved after deserialization"
    
    os.remove(SERIALIZED_PATH)  
