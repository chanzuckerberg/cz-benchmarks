from unittest.mock import patch

import pytest
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism
from scvi.model import SCVI


@pytest.fixture
def example_dataset():
    dataset_path = "tests/assets/example-small.h5ad"
    data = SingleCellDataset(path=dataset_path, organism=Organism.HUMAN)
    data.load_data()
    return data


def test_parse_args():
    model = SCVI()
    model.parse_args()


def test_run(example_dataset):
    model = SCVI()
    with patch.object(model, "download_model_weights") as mock_download, patch.object(
        model, "run_model"
    ) as mock_run_model, patch.object(
        model.dataset_type, "deserialize", return_value=example_dataset
    ) as mock_deserialize, patch.object(
        example_dataset, "load_data"
    ) as mock_load, patch.object(
        example_dataset, "validate"
    ) as mock_validate, patch.object(
        example_dataset, "unload_data"
    ) as mock_unload, patch.object(
        example_dataset, "serialize"
    ) as mock_serialize:
        model.run()
        mock_deserialize.assert_called_once()
        mock_load.assert_called_once()
        mock_validate.assert_called_once()
        mock_download.assert_called_once()
        mock_run_model.assert_called_once()
        mock_unload.assert_called_once()
        mock_serialize.assert_called_once()
