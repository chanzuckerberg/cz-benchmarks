from czbenchmarks.datasets import utils

import pytest
from czbenchmarks.datasets.utils import load_dataset
from unittest.mock import patch


def test_list_available_datasets():
    """Test that list_available_datasets returns a sorted list of dataset names."""
    # Get the list of available datasets
    datasets = utils.list_available_datasets()

    # Verify it's a list
    assert isinstance(datasets, list)

    # Verify it's not empty
    assert len(datasets) > 0

    # Verify it's sorted alphabetically
    assert datasets == sorted(datasets)

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
