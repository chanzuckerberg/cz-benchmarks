import hydra
import pytest
from czbenchmarks.datasets.types import Organism
from czbenchmarks.utils import load_custom_config
from pathlib import Path
from omegaconf import OmegaConf
from czbenchmarks.utils import initialize_hydra, import_class_from_config


# Sample test class for import testing
class ImportTestClass:
    def __init__(self, param1: str, param2: int):
        self.param1 = param1
        self.param2 = param2


def test_initialize_hydra():
    """Test hydra initialization with default and custom config paths."""
    # Test with default config path
    initialize_hydra()
    assert hydra.core.global_hydra.GlobalHydra.instance().is_initialized()

    # Clear hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Test with custom config path -- hydra requires relative paths
    this_dir = Path(__file__).parent
    custom_path = Path(this_dir / "conf").relative_to(this_dir)
    initialize_hydra(str(custom_path))
    assert hydra.core.global_hydra.GlobalHydra.instance().is_initialized()

    # Clean up
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def test_import_class_from_config(tmp_path):
    """Test importing a class from a configuration file."""
    # Create a temporary config file
    config = {
        "_target_": "tests.test_utils.ImportTestClass",
        "param1": "test",
        "param2": 42,
    }

    config_path = tmp_path / "test_config.yaml"
    OmegaConf.save(config=config, f=config_path)

    # Import the class
    imported_class = import_class_from_config(str(config_path))

    # Verify it's the correct class
    assert imported_class == ImportTestClass

    # Test that we can instantiate it with the config parameters
    instance = imported_class(param1="test", param2=42)
    assert instance.param1 == "test"
    assert instance.param2 == 42


@pytest.mark.parametrize(
    "dataset_name, custom_dataset_config, custom_yaml_content",
    [
        # Change existing default value and add a new key
        (
            "replogle_k562_essential_perturbpredict",
            {
                "_target_": "czbenchmarks.datasets.SingleCellPerturbationDataset",
                "path": "s3://cz-benchmarks-data/datasets/v2/perturb/single_cell/replogle_k562_essential_perturbpredict_de_results_control_cells_v2.h5ad",
                "organism": Organism.HUMAN,
                # change existing default value and add a new key
                "percent_genes_to_mask": 0.075,
            },
            {
                "datasets": {
                    "replogle_k562_essential_perturbpredict": {
                        "path": "/single_cell/replogle_k562_essential_perturbpredict_de_results_control_cells_v2.h5ad",
                        "de_gene_col": "gene_symbol",
                        "key_added": "yes",
                    }
                }
            },
        ),
        # Brand new dataset with custom keys
        (
            "my_dummy_dataset",
            {
                "_target_": "czbenchmarks.datasets.dummy.DummyDataset",
                "path": "/dummy.h5ad",
                "organism": Organism.HUMAN,
                "foo": "bar",
            },
            {
                "datasets": {
                    "my_dummy_dataset": {
                        "_target_": "czbenchmarks.datasets.dummy.DummyDataset",
                        "path": "/dummy.h5ad",
                        "key_added": True,
                    }
                }
            },
        ),
        # Dict-only example - change existing default and add new key
        (
            "tsv2_bladder",
            {
                "path": "/only_dict.h5ad",
                "organism": Organism.MOUSE,  # change default organism
                "dict_only": "yes",  # new key
            },
            None,
        ),
        # YAML-only example with new dataset
        (
            "yaml_only_dataset",
            None,
            {
                "datasets": {
                    "yaml_only_dataset": {
                        "_target_": "czbenchmarks.datasets.dummy.DummyDataset",
                        "path": "/yaml_only_dataset.h5ad",
                        "organism": Organism.HUMAN,
                    }
                }
            },
        ),
    ],
)
def test_load_custom_config(
    tmp_path, dataset_name, custom_dataset_config, custom_yaml_content
):
    """Test load_custom_config supports both YAML path and dict updates, including changing existing defaults and adding new keys."""

    # Prepare YAML file from parameterized content
    custom_yaml_path = None
    if custom_yaml_content:
        custom_yaml_path = tmp_path / "custom_config.yaml"
        OmegaConf.save(config=custom_yaml_content, f=custom_yaml_path)
        custom_yaml_path = str(custom_yaml_path)

    custom_cfg = load_custom_config(
        item_name=dataset_name,
        config_name="datasets",
        custom_config_path=custom_yaml_path,
        class_update_kwargs=custom_dataset_config,
    )

    # All dict-provided keys should be present and match
    if custom_dataset_config:
        for key, value in custom_dataset_config.items():
            if key == "organism":
                assert str(custom_cfg[key]) == str(value)
            else:
                assert custom_cfg[key] == value

    # YAML-provided keys for this item should be present; when overlapping, dict wins
    if custom_yaml_content:
        yaml_items = custom_yaml_content.get("datasets", {}).get(dataset_name, {})
        if yaml_items:
            for key, yaml_value in yaml_items.items():
                expected_value = (
                    custom_dataset_config.get(key, yaml_value)
                    if custom_dataset_config
                    else yaml_value
                )
                if key == "organism":
                    assert str(custom_cfg[key]) == str(expected_value)
                else:
                    assert custom_cfg[key] == expected_value
