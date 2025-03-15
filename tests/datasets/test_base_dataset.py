import pytest
import pandas as pd
from czbenchmarks.datasets.types import DataType, Organism
from czbenchmarks.models.types import ModelType
from tests.utils import DummyDataset
import numpy as np


# Tests setting an input for the dataset.
def test_set_input(dummy_dataset):
    dummy_dataset.set_input(DataType.ORGANISM, Organism.HUMAN)
    assert dummy_dataset.get_input(DataType.ORGANISM) == Organism.HUMAN


# Tests setting an output for the dataset.
def test_set_output():
    ds = DummyDataset("dummy_path")
    ds.set_output(ModelType.BASELINE, DataType.ORGANISM, Organism.MOUSE)
    assert ds.get_output(ModelType.BASELINE, DataType.ORGANISM) == Organism.MOUSE


# Tests that getting an input fails when the key is missing.
def test_get_input_missing_key():
    ds = DummyDataset("dummy_path")
    with pytest.raises(KeyError):
        ds.get_input(DataType.ANNDATA)


# Tests that getting an output fails when the model is missing.
def test_get_output_missing_model():
    ds = DummyDataset("dummy_path")
    with pytest.raises(KeyError):
        ds.get_output(ModelType.SCVI, DataType.ANNDATA)


# Tests that getting an output fails when the data type is missing.
def test_get_output_missing_data_type():
    ds = DummyDataset("dummy_path")
    ds.set_output(ModelType.BASELINE, DataType.ORGANISM, Organism.HUMAN)
    with pytest.raises(KeyError):
        ds.get_output(ModelType.BASELINE, DataType.ANNDATA)


# Tests that dataset validation passes when the path exists.
def test_validate_dataset_path_exists(tmp_path):
    p = tmp_path / "existing_path"
    p.touch()
    ds = DummyDataset(str(p))
    ds.validate()


# Tests that dataset validation fails when the path does not exist.
def test_validate_dataset_path_not_exists():
    ds = DummyDataset("non_existing_path")
    with pytest.raises(ValueError):
        ds.validate()


# Tests that setting an input fails when the input type is invalid.
def test_validate_input_type():
    ds = DummyDataset("dummy_path")
    ds.set_input(DataType.ORGANISM, Organism.HUMAN)
    ds.set_output(ModelType.BASELINE, DataType.ORGANISM, Organism.HUMAN)
    with pytest.raises(TypeError):
        ds.set_input(DataType.ORGANISM, "not_an_organism")


# Tests that setting an input fails when the input type in a dictionary is invalid.
def test_validate_input_type_with_dict():
    ds = DummyDataset("dummy_path")
    valid_dict = {"test": pd.DataFrame()}
    ds.set_input(DataType.PERTURBATION_TRUTH, valid_dict)
    with pytest.raises(TypeError):
        ds.set_input(DataType.PERTURBATION_TRUTH, {"test": "not_dataframe"})
    with pytest.raises(TypeError):
        ds.set_input(DataType.PERTURBATION_TRUTH, {1: pd.DataFrame()})


# Tests that setting an input with an output type raises an error.
def test_set_input_with_output_type():
    ds = DummyDataset("dummy_path")
    with pytest.raises(ValueError, match="Cannot set output type as input"):
        ds.set_input(DataType.EMBEDDING, np.array([1, 2, 3]))


# Tests that setting an output with an input type raises an error.
def test_set_output_with_input_type():
    ds = DummyDataset("dummy_path")
    with pytest.raises(ValueError, match="Cannot set input type as output"):
        ds.set_output(ModelType.BASELINE, DataType.ORGANISM, Organism.HUMAN)


# Tests that setting an output with wrong value type raises an error.
def test_set_output_wrong_value_type():
    ds = DummyDataset("dummy_path")
    with pytest.raises(TypeError):
        ds.set_output(ModelType.BASELINE, DataType.EMBEDDING, "not_an_array")


# Tests the serialization and deserialization of the dataset.
def test_serialize_deserialize(tmp_path, dummy_dataset):
    path = tmp_path / "serialized.dill"
    dummy_dataset.serialize(str(path))
    loaded = DummyDataset.deserialize(str(path))
    assert isinstance(loaded, type(dummy_dataset))
    assert (
        loaded.get_input(DataType.ANNDATA).shape
        == dummy_dataset.get_input(DataType.ANNDATA).shape
    )
