import sys
from czbenchmarks.datasets import utils
import types
from czbenchmarks.datasets.types import Organism
from czbenchmarks.datasets.utils import load_dataset, load_customized_dataset
from unittest.mock import patch
import pytest


# FIXME MICHELLE find class with data that loads more quickly
@pytest.mark.parametrize(
    "dataset_path, dataset_name, custom_dataset_config",
    [
        # (
        #     "s3://cz-benchmarks-data/datasets/v2/perturb/single_cell/replogle_k562_essential_perturbpredict_de_results_control_cells_v2.h5ad",
        #     "replogle_k562_essential_perturbpredict",
        #     {"_target_": "czbenchmarks.datasets.SingleCellPerturbationDataset", "organism": Organism.HUMAN, "percent_genes_to_mask": .075},
        # ),
        (
            "dummy.h5ad",
            "my_dummy_dataset",
            {
                "_target_": "czbenchmarks.datasets.dummy.DummyDataset",
                "organism": Organism.HUMAN,
                "foo": "bar",
            },
        ),
    ],
)
def test_load_customized_dataset(
    tmp_path, dataset_path, dataset_name, custom_dataset_config
):
    """Test load_customized_dataset instantiates and loads a customized dataset."""

    if not dataset_path.startswith("s3://"):
        # Create a dummy file to represent the dataset
        dummy_file = tmp_path / dataset_path
        dummy_file.write_text("dummy content")
        custom_dataset_config["path"] = str(dummy_file)
    else:
        custom_dataset_config["path"] = dataset_path

    # Create a dummy dataset class
    class DummyDataset:
        def __init__(self, path, organism, **kwargs):
            self.path = path
            self.organism = organism
            self.kwargs = kwargs
            self.loaded = False

        def load_data(self):
            self.loaded = True

    # Dynamically create a dummy module and add DummyDataset to it
    target_module = custom_dataset_config["_target_"].split(".")[:-1]
    target_module = ".".join(target_module)
    dummy_module = types.ModuleType(target_module)
    dummy_module.DummyDataset = DummyDataset
    sys.modules[target_module] = dummy_module

    # Call load_customized_dataset with the dummy class
    dataset = load_customized_dataset(
        dataset_name=dataset_name, custom_dataset_config=custom_dataset_config
    )

    assert isinstance(dataset, DummyDataset)
    assert dataset.loaded is True
    assert dataset.path == custom_dataset_config["path"]
    assert dataset.organism == custom_dataset_config["organism"]
    assert dataset.kwargs["foo"] == custom_dataset_config["foo"]


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
            "url": "s3://cz-benchmarks-data/datasets/v2/perturb/single_cell/replogle_k562_essential_perturbpredict_de_results_control_cells_v2.h5ad",
        },
        "tsv2_bladder": {
            "organism": "homo_sapiens",
            "url": "s3://cz-benchmarks-data/datasets/v1/cell_atlases/Homo_sapiens/Tabula_Sapiens_v2/homo_sapiens_10df7690-6d10-4029-a47e-0f071bb2df83_Bladder_v2_curated.h5ad",
        },
    }
    assert (
        datasets["replogle_k562_essential_perturbpredict"]
        == expected_datasets["replogle_k562_essential_perturbpredict"]
    )
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
