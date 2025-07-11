import os
import hydra
from hydra.utils import instantiate
from typing import List, Optional
import yaml
from omegaconf import OmegaConf
from czbenchmarks.utils import initialize_hydra
from czbenchmarks.file_utils import download_file_from_remote
from .base import BaseDataset
import logging

log = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str,
    config_path: Optional[str] = None,
) -> BaseDataset:
    """
    Loads, downloads (if needed), and instantiates a dataset using Hydra configuration.

    Args:
        dataset_name (str): Name of the dataset as specified in the configuration.
        config_path (Optional[str]): Optional path to a custom config YAML file. If not provided,
            only the package's default config is used.

    Returns:
        BaseDataset: Instantiated dataset object with data loaded.

    Raises:
        FileNotFoundError: If the custom config file does not exist.
        ValueError: If the specified dataset is not found in the configuration.

    Notes:
        - Merges custom config with default config if provided.
        - Downloads dataset file if remote path is specified.
        - Uses Hydra for instantiation and configuration management.
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
    dataset_path = download_file_from_remote(dataset_info["path"])

    # Instantiate the dataset using Hydra
    dataset = instantiate(dataset_info)
    dataset.path = dataset_path  # os.path.expanduser(dataset.path)

    # Load the dataset into memory
    dataset.load_data()

    return dataset


def list_available_datasets() -> List[str]:
    """
    Returns a sorted list of all dataset names defined in the datasets.yaml Hydra configuration.

    Returns:
        List[str]: Alphabetically sorted list of available dataset names.

    Notes:
        - Loads configuration using Hydra.
        - Extracts dataset names from the 'datasets' section of the config.
    """
    initialize_hydra()

    # Load the datasets configuration
    cfg = OmegaConf.to_container(hydra.compose(config_name="datasets"), resolve=True)

    # Extract dataset names
    dataset_names = list(cfg.get("datasets", {}).keys())

    # Sort alphabetically for easier reading
    dataset_names.sort()

    return dataset_names
