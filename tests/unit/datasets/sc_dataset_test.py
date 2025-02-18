import os
import pytest
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path
from typing import Any

from czibench.datasets.base import BaseDataset
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism
from czibench.datasets.utils import load_dataset  # If you want to test load_dataset
from omegaconf import OmegaConf


TEST_H5AD_PATH = "tests/assets/example-small.h5ad"

class DummyDataset(BaseDataset):
    def _validate(self) -> None:
        # Here we just simulate a "valid" dataset if a path is set.
        if not hasattr(self, "path"):
            raise ValueError("DummyDataset: path attribute is missing.")

    def load_data(self) -> None:
        # No real data loading; just a flag for test demonstration.
        self.data_loaded = True

    def unload_data(self) -> None:
        # Simulate unloading data by removing the flag.
        self.data_loaded = False


@pytest.fixture
def sc_dataset():
    dataset = SingleCellDataset(path=TEST_H5AD_PATH, organism=Organism.HUMAN)
    dataset.load_data()
    yield dataset
    dataset.unload_data()


@pytest.fixture
def dummy_dataset(tmp_path):
    dummy_path = tmp_path / "dummy_file.txt"
    dummy_path.write_text("Just a dummy file.")
    dataset = DummyDataset(path=str(dummy_path))
    return dataset


@pytest.fixture
def invalid_anndata_file(tmp_path):
    # Create a small AnnData object
    obs_data = pd.DataFrame({"cell_type": ["B", "T"]}, index=["cell1", "cell2"])
    var_data = pd.DataFrame(
        {"gene_symbol": ["XYZ0001", "XYZ0002"]},  # Notice: no 'ensembl_id' column
        index=["XYZ0001", "XYZ0002"]
    )
    X = np.random.rand(2, 2)
    adata = ad.AnnData(X=X, obs=obs_data, var=var_data)

    # Write to temporary file
    invalid_path = tmp_path / "invalid_anndata.h5ad"
    adata.write_h5ad(invalid_path)
    return invalid_path


def test_dummy_dataset_validates(dummy_dataset):
    dummy_dataset.validate()
    dummy_dataset.load_data()
    assert dummy_dataset.data_loaded is True


def test_dummy_dataset_missing_path():
    dataset = DummyDataset(path="non_existing_file.txt")
    with pytest.raises(ValueError, match="Dataset non_existing_file.txt is not valid"):
        dataset.validate()


def test_dummy_dataset_serialize_deserialize(dummy_dataset, tmp_path):
    dummy_dataset.validate()
    dummy_dataset.load_data()

    out_file = tmp_path / "dummy_dataset_serialized.dill"
    dummy_dataset.serialize(str(out_file))
    assert out_file.exists()

    loaded_dataset = BaseDataset.deserialize(str(out_file))
    assert isinstance(loaded_dataset, DummyDataset)
    # Now we expect data_loaded = True because dill rehydrates the entire object:
    assert getattr(loaded_dataset, "data_loaded", False), (
        "After deserialization, data_loaded should remain True "
        "because that's how dill rehydrates the object."
    )


def test_load_data(sc_dataset):
    assert sc_dataset.adata is not None, "AnnData object should be loaded"
    assert isinstance(sc_dataset.sample_metadata, pd.DataFrame), "Metadata should be a DataFrame"
    assert not sc_dataset.sample_metadata.empty, "Metadata should not be empty"


def test_unload_data(sc_dataset):
    sc_dataset.unload_data()
    assert sc_dataset.adata is None, "AnnData object should be None after unload"


def test_validate_success(sc_dataset):
    sc_dataset.validate()


def test_sc_dataset_serialize_deserialize(sc_dataset, tmp_path):

    out_file = tmp_path / "sc_dataset_serialized.dill"
    sc_dataset.serialize(str(out_file))
    assert out_file.exists()

    loaded_sc_dataset = SingleCellDataset.deserialize(str(out_file))
    assert isinstance(loaded_sc_dataset, SingleCellDataset)


def test_validate_invalid_organism():
    dataset = SingleCellDataset(path=TEST_H5AD_PATH, organism="INVALID")
    with pytest.raises(ValueError, match="Dataset does not contain anndata object"):
        dataset.validate()


def test_validate_invalid_gene_prefix(invalid_anndata_file):
    dataset = SingleCellDataset(path=str(invalid_anndata_file), organism=Organism.HUMAN)
    dataset.load_data()
    with pytest.raises(ValueError, match="Dataset does not contain valid gene names"):
        dataset.validate()


def test_organism_enum():
    assert Organism.from_string("human") == Organism.HUMAN
    assert Organism.from_string("HOMO_SAPIENS") == Organism.HUMAN
    assert Organism.from_string("mouse") == Organism.MOUSE
    assert Organism.from_string("mus_musculus") == Organism.MOUSE

    with pytest.raises(ValueError, match="Unknown organism"):
        Organism.from_string("unknown_organism")


@pytest.mark.skip(reason="Requires full Hydra config setup for 'datasets' to test properly.")
def test_load_dataset_hydra():
    cfg = OmegaConf.create(
        {
            "datasets": {
                "test_h5ad": {
                    "_target_": "czibench.datasets.sc.SingleCellDataset",
                    "path": TEST_H5AD_PATH,
                    "organism": "${organism:human}"
                }
            }
        }
    )
    dataset = SingleCellDataset(path=TEST_H5AD_PATH, organism=Organism.HUMAN)
    dataset.load_data()
    assert dataset.adata is not None
    dataset.unload_data()


