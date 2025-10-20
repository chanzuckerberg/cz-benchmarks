import sys
import types
from urllib.parse import urlparse
from pathlib import Path
from omegaconf import OmegaConf
import scanpy as sc
from czbenchmarks.utils import load_custom_config
from czbenchmarks.datasets.types import Organism
from czbenchmarks.datasets.utils import (
    list_available_datasets,
    load_dataset,
    load_custom_dataset,
)
from unittest.mock import Mock, patch
import pytest
from tests.utils import create_dummy_anndata


@pytest.fixture()
def adata():
    adata = create_dummy_anndata(
        n_cells=5,
        n_genes=3,
        obs_columns=["cell_type"],
        organism=Organism.HUMAN,
    )
    return adata


def test_list_available_datasets():
    """Test that list_available_datasets returns a sorted list of dataset names."""
    # Get the list of available datasets
    datasets = list_available_datasets()

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


@pytest.mark.parametrize(
    "dataset_path, dataset_name, custom_dataset_config, custom_yaml_content",
    [
        (  # dictionary configuration
            "/extra_directory/extra_subdirectory/mouse_test.h5ad",
            "tsv2_bladder",
            {
                "_target_": "czbenchmarks.datasets.SingleCellLabeledDataset",
                "path": "__DUMMY__",
                "organism": Organism.MOUSE,
            },
            None,
        ),
        (  # YAML configuration
            "human_test.h5ad",
            "tsv2_bladder",
            None,
            {
                "datasets": {
                    "tsv2_bladder": {
                        "_target_": "czbenchmarks.datasets.SingleCellLabeledDataset",
                        "organism": Organism.HUMAN,
                    }
                }
            },
        ),
        (  # YAML plus dict; dict should override YAML path
            "gorilla_test.h5ad",
            "tsv2_bladder",
            {
                "_target_": "czbenchmarks.datasets.SingleCellLabeledDataset",
                "organism": Organism.GORILLA,
                "path": "__DUMMY__",
            },
            {
                "datasets": {
                    "tsv2_bladder": {
                        "_target_": "czbenchmarks.datasets.SingleCellLabeledDataset",
                        "organism": "__DUMMY__",
                        "path": "__DUMMY__",
                    }
                }
            },
        ),
        (  # New dataset with dummy class
            "dummy.h5ad",
            "my_dummy_dataset",
            {
                "_target_": "czbenchmarks.datasets.dummy.DummyDataset",
                "organism": Organism.HUMAN,
                "foo": "bar",
            },
            None,
        ),
        pytest.param(  # Valid s3 file configuration, must be downloaded
            "s3://cz-benchmarks-data/datasets/v1/cell_atlases/Homo_sapiens/Tabula_Sapiens_v2/homo_sapiens_10df7690-6d10-4029-a47e-0f071bb2df83_Bladder_v2_curated.h5ad",
            "tsv2_bladder",
            {
                "_target_": "czbenchmarks.datasets.SingleCellLabeledDataset",
                "path": "__DUMMY__",
            },
            None,
            marks=pytest.mark.integration,
        ),
    ],
)
def test_load_custom_dataset(
    tmp_path,
    adata,
    dataset_path,
    dataset_name,
    custom_dataset_config,
    custom_yaml_content,
):
    """Test load_custom_dataset instantiates and loads a customized dataset."""

    protocol = urlparse(str(dataset_path)).scheme
    if not protocol:
        # Create a dummy file to represent the dataset
        # Normalize dataset_path to be under tmp_path even if it starts with '/'
        normalized_dataset_path = dataset_path.lstrip("/")
        dummy_file = tmp_path / normalized_dataset_path
        if dummy_file.parent:
            dummy_file.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(dummy_file)
        dataset_path = str(dummy_file)

    # Put file path in dataset config if it exists since it overrides the YAML data
    if custom_dataset_config:
        custom_dataset_config["path"] = dataset_path
    else:
        custom_yaml_content["datasets"][dataset_name]["path"] = dataset_path

    # Prepare optional YAML file and update custom config paths
    custom_yaml_path = None
    if custom_yaml_content:
        custom_yaml_path = tmp_path / "custom_config.yaml"
        OmegaConf.save(config=custom_yaml_content, f=custom_yaml_path)
        custom_yaml_path = str(custom_yaml_path)

    # Create a dummy dataset class and module if needed
    if dataset_name == "my_dummy_dataset":

        class DummyDataset:
            def __init__(self, path, organism, **kwargs):
                self.path = path
                self.organism = organism
                self.kwargs = kwargs

            def load_data(self):
                self.adata = sc.read_h5ad(self.path)

        target_module = custom_dataset_config["_target_"].split(".")[:-1]
        target_module = ".".join(target_module)
        dummy_module = types.ModuleType(target_module)
        dummy_module.DummyDataset = DummyDataset
        sys.modules[target_module] = dummy_module

    # Call load_custom_dataset with the dummy class
    dataset = load_custom_dataset(
        dataset_name=dataset_name,
        custom_dataset_config_path=custom_yaml_path,
        custom_dataset_kwargs=custom_dataset_config,
        cache_dir=tmp_path if protocol else None,
    )

    expected_cfg = load_custom_config(
        item_name=dataset_name,
        config_name="datasets",
        custom_config_path=custom_yaml_path,
        class_update_kwargs=custom_dataset_config,
    )
    if protocol:
        download_path = Path(dataset_path).name
        download_path = tmp_path / download_path
        expected_cfg["path"] = str(download_path)

    assert dataset.adata is not None

    for key in expected_cfg:
        if key == "_target_":
            target = expected_cfg[key]
            cls_name = dataset.__class__.__name__
            assert cls_name == target.split(".")[-1]
        elif key == "path":
            if not protocol:
                assert str(getattr(dataset, key)) == str(expected_cfg[key])
        elif key == "organism":
            assert str(getattr(dataset, key)) == str(expected_cfg[key])
        else:
            if hasattr(dataset, "kwargs") and dataset.kwargs and key in dataset.kwargs:
                assert dataset.kwargs[key] == expected_cfg[key]
            else:
                assert getattr(dataset, key) == expected_cfg[key]


class TestUtils:
    """Extended tests for utils.py."""

    @patch("czbenchmarks.datasets.utils.instantiate")
    @patch("czbenchmarks.datasets.utils.hydra.compose")
    @patch("czbenchmarks.datasets.utils.download_file_from_remote")
    @patch("czbenchmarks.datasets.utils.initialize_hydra")
    def test_load_dataset_happy_path(
        self,
        mock_initialize_hydra,
        mock_download,
        mock_compose,
        mock_instantiate,
    ):
        """Test that load_dataset successfully loads a configured dataset."""
        dataset_name = "tsv2_bladder"
        remote_path = "s3://example/dataset.h5ad"
        dataset_config = {
            "_target_": "czbenchmarks.datasets.SingleCellLabeledDataset",
            "organism": Organism.HUMAN,
            "path": remote_path,
        }

        mock_compose.return_value = types.SimpleNamespace(
            datasets={dataset_name: dataset_config}
        )
        mock_download.return_value = "/tmp/local_dataset.h5ad"

        mock_dataset = types.SimpleNamespace()
        mock_dataset.load_data = Mock()
        mock_instantiate.return_value = mock_dataset

        loaded_dataset = load_dataset(dataset_name)

        mock_initialize_hydra.assert_called_once()
        mock_compose.assert_called_once_with(config_name="datasets")
        mock_download.assert_called_once_with(remote_path)
        mock_instantiate.assert_called_once_with(dataset_config)
        mock_dataset.load_data.assert_called_once()
        assert loaded_dataset is mock_dataset

    def test_load_dataset_invalid_name(self):
        """Test that load_dataset raises ValueError for invalid dataset name."""
        with pytest.raises(ValueError):
            load_dataset("invalid_dataset")
