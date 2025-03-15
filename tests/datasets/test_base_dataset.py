import pytest
import pandas as pd
from czbenchmarks.datasets.types import DataType, Organism
from czbenchmarks.models.types import ModelType
from tests.utils import DummyDataset
import numpy as np


def test_set_input(dummy_dataset):
    """Tests setting an input for the dataset."""
    dummy_dataset.set_input(DataType.ORGANISM, Organism.HUMAN)
    assert dummy_dataset.get_input(DataType.ORGANISM) == Organism.HUMAN


def test_set_output():
    """Tests setting an output for the dataset."""
    ds = DummyDataset("dummy_path")
    ds.set_output(ModelType.BASELINE, DataType.EMBEDDING, np.array([1, 2, 3]))
    assert np.all(
        ds.get_output(ModelType.BASELINE, DataType.EMBEDDING) == np.array([1, 2, 3])
    )


def test_get_input_missing_key():
    """Tests that getting an input fails when the key is missing."""
    ds = DummyDataset("dummy_path")
    with pytest.raises(KeyError):
        ds.get_input(DataType.ANNDATA)


def test_get_output_missing_model():
    """Tests that getting an output fails when the model is missing."""
    ds = DummyDataset("dummy_path")
    with pytest.raises(KeyError):
        ds.get_output(ModelType.SCVI, DataType.ANNDATA)


def test_get_output_missing_data_type():
    """Tests that getting an output fails when the data type is missing."""
    ds = DummyDataset("dummy_path")
    with pytest.raises(KeyError):
        ds.get_output(ModelType.BASELINE, DataType.ANNDATA)


def test_validate_dataset_path_exists(dummy_human_anndata):
    """Test that validation succeeds when dataset path exists."""
    dummy_human_anndata.validate()  # Should not raise any exceptions


def test_validate_dataset_path_not_exists(dummy_human_anndata):
    """Test that validation fails when dataset path does not exist."""
    dummy_human_anndata.path = "non_existing_path.h5ad"
    with pytest.raises(ValueError, match="Dataset path does not exist"):
        dummy_human_anndata.validate()


def test_validate_input_type():
    """Tests that setting an input fails when the input type is invalid."""
    ds = DummyDataset("dummy_path")
    ds.set_input(DataType.ORGANISM, Organism.HUMAN)
    with pytest.raises(TypeError):
        ds.set_input(DataType.ORGANISM, "not_an_organism")


def test_validate_input_type_with_dict():
    """Tests that setting an input fails when the input type in a dictionary is invalid."""
    ds = DummyDataset("dummy_path")
    valid_dict = {"test": pd.DataFrame()}
    ds.set_input(DataType.PERTURBATION_TRUTH, valid_dict)
    with pytest.raises(TypeError):
        ds.set_input(DataType.PERTURBATION_TRUTH, {"test": "not_dataframe"})
    with pytest.raises(TypeError):
        ds.set_input(DataType.PERTURBATION_TRUTH, {1: pd.DataFrame()})


def test_set_input_with_output_type():
    """Tests that setting an input with an output type raises an error."""
    ds = DummyDataset("dummy_path")
    with pytest.raises(ValueError, match="Cannot set output type as input.*"):
        ds.set_input(DataType.EMBEDDING, np.array([1, 2, 3]))


def test_set_output_with_input_type():
    """Tests that setting an output with an input type raises an error."""
    ds = DummyDataset("dummy_path")
    with pytest.raises(ValueError, match="Cannot set input type as output.*"):
        ds.set_output(ModelType.BASELINE, DataType.ORGANISM, Organism.HUMAN)


def test_set_output_wrong_value_type():
    """Tests that setting an output with wrong value type raises an error."""
    ds = DummyDataset("dummy_path")
    with pytest.raises(TypeError):
        ds.set_output(ModelType.BASELINE, DataType.EMBEDDING, "not_an_array")


def test_serialize_deserialize(tmp_path, dummy_dataset):
    """Tests the serialization and deserialization of the dataset."""
    path = tmp_path / "serialized.dill"
    dummy_dataset.serialize(str(path))
    loaded = DummyDataset.deserialize(str(path))
    assert isinstance(loaded, type(dummy_dataset))
    assert (
        loaded.get_input(DataType.ANNDATA).shape
        == dummy_dataset.get_input(DataType.ANNDATA).shape
    )
