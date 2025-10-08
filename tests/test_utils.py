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
    "dataset_path, dataset_name, custom_dataset_config",
    [
        (
            "dummy.h5ad",
            "my_dummy_dataset",
            {
                "_target_": "czbenchmarks.datasets.dummy.DummyDataset",
                "organism": Organism.HUMAN,
                "foo": "bar",
            },
        ),
        (
            "s3://cz-benchmarks-data/datasets/v2/perturb/single_cell/replogle_k562_essential_perturbpredict_de_results_control_cells_v2.h5ad",
            "replogle_k562_essential_perturbpredict",
            {
                "_target_": "czbenchmarks.datasets.SingleCellPerturbationDataset",
                "organism": Organism.HUMAN,
                "percent_genes_to_mask": 0.075,
            },
        ),
    ],
)
def test_load_custom_config(dataset_path, dataset_name, custom_dataset_config):
    """Test load_customized_config instantiates and loads a customized configuration."""

    custom_dataset_config["path"] = dataset_path
    custom_cfg = load_custom_config(
        item_name=dataset_name,
        config_name="datasets",
        class_update_kwargs=custom_dataset_config,
    )

    assert custom_cfg.path == custom_dataset_config["path"]
    for key, value in custom_dataset_config.items():
        assert custom_cfg[key] == value


if __name__ == "__main__":
    pytest.main(["-v", __file__, "-k", "test_load_custom_config"])
