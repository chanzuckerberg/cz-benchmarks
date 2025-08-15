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
        "replogle_k562_essential_perturbpredict": {
            "organism": "homo_sapiens",
            "url": "s3://cz-benchmarks-data/datasets/v1/perturb/single_cell/replogle_k562_essential_perturbpredict_de_results_control_cells.h5ad",
        },
        "tsv2_bladder": {
            "organism": "homo_sapiens",
            "url": "s3://cz-benchmarks-data/datasets/v1/cell_atlases/Homo_sapiens/Tabula_Sapiens_v2/homo_sapiens_10df7690-6d10-4029-a47e-0f071bb2df83_Bladder_v2_curated.h5ad",
        },
    }
    assert datasets["replogle_k562_essential_perturbpredict"] == expected_datasets["replogle_k562_essential_perturbpredict"]
    assert datasets["tsv2_bladder"] == expected_datasets["tsv2_bladder"]
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
