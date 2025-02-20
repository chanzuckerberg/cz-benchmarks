from unittest.mock import MagicMock, patch

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from czibench.datasets.base import BaseDataset
from czibench.datasets.types import DataType
from czibench.models.base import BaseModelImplementation


class TestDataset(BaseDataset):
    def _validate(self):
        pass

    def load_data(self):
        self._loaded = True

    def unload_data(self):
        self._loaded = False


class TestModel(BaseModelImplementation):
    dataset_type = TestDataset

    @property
    def inputs(self):
        return {DataType.ANNDATA}

    @property
    def outputs(self):
        return {DataType.EMBEDDING}

    def _validate_dataset(self, dataset: TestDataset):
        pass

    def get_model_weights_subdir(self) -> str:
        return "test_subdir"

    def _download_model_weights(self):
        pass

    def run_model(self):
        self.data.set_output(DataType.EMBEDDING, np.array([[1, 2], [3, 4]]))

    def parse_args(self):
        pass


@pytest.fixture
def base_dataset():
    ds = TestDataset(path="tests/assets/example-small.h5ad")
    ds.set_input(DataType.ANNDATA, ad.AnnData(X=np.array([[1, 2], [3, 4]])))
    ds.set_output(DataType.METADATA, pd.DataFrame({"test": [1, 2, 3]}))
    return ds


@pytest.fixture
def test_model():
    return TestModel()


def test_validate_dataset(test_model, base_dataset):
    test_model.validate_dataset(base_dataset)


def test_validate_dataset_type_mismatch(test_model):
    class DummyDataset(BaseDataset):
        def _validate(self):
            pass

        def load_data(self):
            pass

        def unload_data(self):
            pass

    dummy = DummyDataset(path="dummy")
    with pytest.raises(ValueError, match="Dataset type mismatch"):
        test_model.validate_dataset(dummy)


def test_set_output(test_model, base_dataset):
    test_model.data = base_dataset
    value = ad.AnnData(X=np.array([[5, 6], [7, 8]]))
    test_model.set_output(DataType.ANNDATA, value)
    np.testing.assert_array_equal(
        test_model.data.get_output(DataType.ANNDATA).X, value.X
    )


@patch("os.path.exists", return_value=True)
@patch("os.listdir", return_value=[])
def test_download_model_weights(mock_listdir, mock_exists, test_model):
    test_model.get_model_weights_subdir = MagicMock(return_value="test_subdir")
    test_model._download_model_weights = MagicMock()
    test_model.download_model_weights()
    test_model._download_model_weights.assert_called_once()


@patch("czibench.models.base.BaseDataset.deserialize")
def test_run(mock_deserialize, test_model, base_dataset):
    mock_deserialize.return_value = base_dataset
    test_model.run_model = MagicMock()
    test_model.download_model_weights = MagicMock()
    base_dataset.load_data = MagicMock()
    base_dataset.validate = MagicMock()
    base_dataset.unload_data = MagicMock()
    base_dataset.serialize = MagicMock()
    test_model.run()
    test_model.run_model.assert_called_once()


class DummyDataset:
    def load_data(self):
        pass

    def validate(self):
        pass

    def unload_data(self):
        pass

    def serialize(self, path):
        pass


@patch("czibench.models.base.BaseDataset.deserialize", return_value=DummyDataset())
def test_run_invalid_data_type(mock_deserialize, test_model):
    with pytest.raises(ValueError, match="Dataset type mismatch"):
        test_model.run()
