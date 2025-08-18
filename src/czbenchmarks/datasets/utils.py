import os
import hydra
from hydra.utils import instantiate
from typing import Dict, Optional, Literal
import yaml
from omegaconf import OmegaConf
from .dataset import Dataset
from czbenchmarks.utils import initialize_hydra
from czbenchmarks.file_utils import download_file_from_remote
import logging

log = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str,
    backed: Literal['r', 'r+'] | bool | None = None,  # FIXME MICHELLE: for testing, will remove if not used
    config_path: Optional[str] = None,
) -> Dataset:
    """
    Load, download (if needed), and instantiate a dataset using Hydra configuration.

    Args:
        dataset_name (str): Name of the dataset as specified in the configuration.
        backed (Literal['r', 'r+'] | bool | None): Whether to load the dataset into memory 
            (this is the default, None) or use backed mode.
        config_path (Optional[str]): Optional path to a custom config YAML file. If not provided,
            only the package's default config is used.

    Returns:
        Dataset: Instantiated dataset object with data loaded.

    Raises:
        FileNotFoundError: If the custom config file does not exist.
        ValueError: If the specified dataset is not found in the configuration.

    Notes:
        - Merges custom config with default config if provided.
        - Downloads dataset file if a remote path is specified using `download_file_from_remote`.
        - Uses Hydra for instantiation and configuration management.
        - The returned dataset object is an instance of the `Dataset` class or its subclass.
    """
    initialize_hydra()

    # Load default config first and make it unstructured
    cfg = OmegaConf.create(
        OmegaConf.to_container(hydra.compose(config_name="datasets"), resolve=True)
    )

    # If custom config provided, load and merge it
    if config_path is not None:
        # Expand user path (handles ~)
        config_path = os.path.expanduser(config_path)
        config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Custom config file not found: {config_path}")

        # Load custom config
        with open(config_path) as f:
            custom_cfg = OmegaConf.create(yaml.safe_load(f))

        # Merge configs
        cfg = OmegaConf.merge(cfg, custom_cfg)

    if dataset_name not in cfg.datasets:
        raise ValueError(f"Dataset {dataset_name} not found in config")

    dataset_info = cfg.datasets[dataset_name]

    # Handle local caching and remote downloading
    dataset_info["path"] = download_file_from_remote(dataset_info["path"])

    # Instantiate the dataset using Hydra
    dataset = instantiate(dataset_info)

    # Load the dataset into memory
    dataset.load_data(backed=backed)  # FIXME MICHELLE: for testing, will remove if not used

    return dataset


def list_available_datasets() -> Dict[str, Dict[str, str]]:
    """
    Return a sorted list of all dataset names defined in the `datasets.yaml` Hydra configuration.

    Returns:
        List[str]: Alphabetically sorted list of available dataset names.

    Notes:
        - Loads configuration using Hydra.
        - Extracts dataset names from the `datasets` section of the configuration.
        - Sorts the dataset names alphabetically for easier readability.
    """
    initialize_hydra()

    # Load the datasets configuration
    cfg = OmegaConf.to_container(hydra.compose(config_name="datasets"), resolve=True)

    # Extract dataset names
    datasets = {
        name: {
            "organism": str(dataset_info.get("organism", "Unknown")),
            "url": dataset_info.get("path", "Unknown"),
        }
        for name, dataset_info in cfg.get("datasets", {}).items()
    }

    # Sort alphabetically for easier reading
    datasets = dict(sorted(datasets.items()))

    return datasets
