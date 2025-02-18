import pytest
import numpy as np
import pandas as pd
import anndata as ad
import os

from czibench.datasets.base import BaseDataset
from czibench.models.base import BaseModelImplementation
from czibench.models.single_cell import BaseSingleCellValidator
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism


class DummySingleCellModel(BaseModelImplementation, BaseSingleCellValidator):
    dataset_type = SingleCellDataset
    available_organisms = [Organism.HUMAN]
    required_obs_keys = ["cell_type", "some_obs_key"]
    required_var_keys = ["ensembl_id"]

    def get_model_weights_subdir(self):
        return "dummy_subdir"

    def _download_model_weights(self):
        pass

    def run_model(self):
        pass

    def parse_args(self):
        return {}


@pytest.fixture
def valid_single_cell_dataset(tmp_path):
    obs = pd.DataFrame(
        {
            "cell_type": ["A", "B"],
            "some_obs_key": [1, 2],
        },
        index=["cell1", "cell2"],
    )
    var = pd.DataFrame(
        {
            "ensembl_id": ["ENSG000001", "ENSG000002"],
        },
        index=["gene1", "gene2"],
    )
    X = np.random.rand(2, 2)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    example_file = tmp_path / "example.h5ad"
    adata.write_h5ad(example_file)
    dataset = SingleCellDataset(path=str(example_file), organism=Organism.HUMAN)
    return dataset


def test_positive_run(valid_single_cell_dataset):
    model = DummySingleCellModel()
    valid_single_cell_dataset.load_data()
    model.validate_dataset(valid_single_cell_dataset)
    model.download_model_weights()
    model.run_model()
    valid_single_cell_dataset.unload_data()
    assert True


def test_dataset_type_mismatch(valid_single_cell_dataset):
    class AnotherDatasetClass:
        pass

    model = DummySingleCellModel()
    with pytest.raises(ValueError) as exc_info:
        model.validate_dataset(AnotherDatasetClass())
    assert "Dataset type mismatch" in str(exc_info.value)


def test_missing_obs_key(valid_single_cell_dataset):
    valid_single_cell_dataset.load_data()
    valid_single_cell_dataset.adata.obs.drop(columns=["some_obs_key"], inplace=True)
    model = DummySingleCellModel()
    with pytest.raises(ValueError) as exc_info:
        model.validate_dataset(valid_single_cell_dataset)
    assert "Missing required obs keys" in str(exc_info.value)
    valid_single_cell_dataset.unload_data()


def test_missing_var_key(valid_single_cell_dataset):
    valid_single_cell_dataset.load_data()
    valid_single_cell_dataset.adata.var.drop(columns=["ensembl_id"], inplace=True)
    model = DummySingleCellModel()
    with pytest.raises(ValueError) as exc_info:
        model.validate_dataset(valid_single_cell_dataset)
    assert "Missing required var keys" in str(exc_info.value)
    valid_single_cell_dataset.unload_data()


def test_invalid_organism(valid_single_cell_dataset):
    valid_single_cell_dataset.organism = Organism.MOUSE
    valid_single_cell_dataset.load_data()
    model = DummySingleCellModel()
    with pytest.raises(ValueError) as exc_info:
        model.validate_dataset(valid_single_cell_dataset)
    assert "is not supported" in str(exc_info.value)
    valid_single_cell_dataset.unload_data()


class ExampleSingleCellValidator(BaseSingleCellValidator):
    available_organisms = [Organism.HUMAN]
    required_obs_keys = ["perturbation"]
    required_var_keys = ["ensembl_id"]


class ExampleSingleCellModel(BaseModelImplementation, ExampleSingleCellValidator):
    def get_model_weights_subdir(self):
        return "example_model_weights"

    def _download_model_weights(self):
        pass

    def run_model(self):
        pass

    def parse_args(self):
        return {}


@pytest.fixture
def valid_dataset():
    dataset_path = "tests/assets/example-small.h5ad"
    assert os.path.exists(dataset_path)
    dataset = SingleCellDataset(dataset_path, organism=Organism.HUMAN)
    dataset.load_data()
    dataset.adata.obs["perturbation"] = "control"
    return dataset


def test_basesinglecellvalidator_positive(valid_dataset):
    validator = ExampleSingleCellValidator()
    validator.validate_dataset(valid_dataset)


def test_basesinglecellvalidator_negative_missing_obs(valid_dataset):
    valid_dataset.adata.obs.drop(columns=["perturbation"], inplace=True)
    validator = ExampleSingleCellValidator()
    with pytest.raises(ValueError, match="Missing required obs keys"):
        validator.validate_dataset(valid_dataset)


def test_basesinglecellvalidator_negative_invalid_organism():
    dataset_path = "tests/assets/example-small.h5ad"
    dataset = SingleCellDataset(dataset_path, organism=Organism.MOUSE)
    dataset.load_data()
    validator = ExampleSingleCellValidator()
    with pytest.raises(ValueError, match="is not supported"):
        validator.validate_dataset(dataset)


def test_basemodelimplementation_positive(valid_dataset):
    model = ExampleSingleCellModel()
    model.validate_dataset(valid_dataset)


def test_basemodelimplementation_negative_dataset_type():
    class AnotherDataset(BaseDataset):
        def _validate(self):
            pass

        def load_data(self):
            pass

        def unload_data(self):
            pass

    another_dataset = AnotherDataset(path="some_path")
    model = ExampleSingleCellModel()
    with pytest.raises(ValueError, match="Dataset type mismatch"):
        model.validate_dataset(another_dataset)
