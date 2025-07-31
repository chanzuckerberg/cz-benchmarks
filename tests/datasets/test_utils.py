from czbenchmarks.datasets import utils

import pytest
from czbenchmarks.datasets.utils import load_dataset
from unittest.mock import patch


def test_list_available_datasets():
    """Test that list_available_datasets returns a sorted list of dataset names."""
    # Get the list of available datasets
    datasets = utils.list_available_datasets()

    # Verify it's a dict
    assert isinstance(datasets, dict)

    # Verify it's not empty
    assert len(datasets) > 0

    # Verify it's sorted alphabetically
    assert list(datasets.keys()) == sorted(datasets.keys())

    # Verify the dataset names match the expected dataset names
    expected_datasets = {
        "adamson_perturb": {
            "organism": "homo_sapiens",
            "url": "s3://cz-benchmarks-data/datasets/v1/perturb/single_cell/adamson_perturbation.h5ad",
        },
        "dixit_perturb": {
            "organism": "homo_sapiens",
            "url": "s3://cz-benchmarks-data/datasets/v1/perturb/single_cell/dixit_perturbation.h5ad",
        },
    }
    assert datasets["adamson_perturb"] == expected_datasets["adamson_perturb"]
    assert datasets["dixit_perturb"] == expected_datasets["dixit_perturb"]
    # Verify all elements are strings
    assert all(isinstance(dataset, str) for dataset in datasets)

    # Verify no empty strings
    assert all(len(dataset) > 0 for dataset in datasets)


class TestUtils:
    """Extended tests for utils.py."""

    @patch("czbenchmarks.datasets.utils.download_file_from_remote")
    @patch("czbenchmarks.datasets.utils.initialize_hydra")
    def test_load_dataset_missing_config(self, mock_initialize_hydra, mock_download):
        """Test that load_dataset raises FileNotFoundError for missing config."""
        with pytest.raises(FileNotFoundError):
            load_dataset("non_existent_dataset", config_path="missing_config.yaml")

    @patch("czbenchmarks.datasets.utils.download_file_from_remote")
    @patch("czbenchmarks.datasets.utils.initialize_hydra")
    def test_load_dataset_invalid_name(self, mock_initialize_hydra, mock_download):
        """Test that load_dataset raises ValueError for invalid dataset name."""
        with pytest.raises(ValueError):
            load_dataset("invalid_dataset")
