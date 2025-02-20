import os
from unittest.mock import patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from czibench.datasets.base import BaseDataset
from czibench.datasets.types import DataType


class TestDataset(BaseDataset):
    def _validate(self):
        pass

    def load_data(self):
        pass

    def unload_data(self):
        pass


@pytest.fixture
def base_dataset():
    return TestDataset(path="tests/assets/example-small.h5ad")


def test_init(base_dataset):
    assert base_dataset.path == "tests/assets/example-small.h5ad"
    assert base_dataset.kwargs == {}


def test_set_input(base_dataset):
    data_type = DataType.ANNDATA
    value = ad.AnnData(X=np.array([[1, 2], [3, 4]]))
    base_dataset.set_input(data_type, value)
    assert (base_dataset.get_input(data_type).X == value.X).all()


def test_set_output(base_dataset):
    data_type = DataType.METADATA
    value = pd.DataFrame({"test": [1, 2, 3]})
    base_dataset.set_output(data_type, value)
    pd.testing.assert_frame_equal(base_dataset.get_output(data_type), value)


def test_set_input_type_error(base_dataset):
    with pytest.raises(TypeError):
        base_dataset.set_input(DataType.ANNDATA, 123)


def test_set_output_type_error(base_dataset):
    with pytest.raises(TypeError):
        base_dataset.set_output(DataType.METADATA, 123)


def test_get_input_key_error(base_dataset):
    with pytest.raises(KeyError):
        base_dataset.get_input(DataType.METADATA)


def test_get_output_key_error(base_dataset):
    with pytest.raises(KeyError):
        base_dataset.get_output(DataType.ANNDATA)


@patch("os.path.exists")
def test_validate_valid(mock_exists, base_dataset):
    mock_exists.return_value = True
    base_dataset.set_input(DataType.ANNDATA, ad.AnnData(X=np.array([[1, 2], [3, 4]])))
    base_dataset.set_output(DataType.METADATA, pd.DataFrame({"test": [1, 2, 3]}))
    base_dataset.validate()


@patch("os.path.exists")
def test_validate_invalid(mock_exists, base_dataset):
    mock_exists.return_value = False
    with pytest.raises(ValueError):
        base_dataset.validate()


def test_serialize(base_dataset):
    base_dataset.serialize("tests/assets/test_output.dill")
    assert os.path.exists("tests/assets/test_output.dill")


def test_deserialize(base_dataset):
    base_dataset.serialize("tests/assets/test_output.dill")
    deserialized_dataset = BaseDataset.deserialize("tests/assets/test_output.dill")
    assert deserialized_dataset.path == "tests/assets/example-small.h5ad"
