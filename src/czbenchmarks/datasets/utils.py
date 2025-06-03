import os
import hydra
from hydra.utils import instantiate
from typing import List, Optional
import yaml
from omegaconf import OmegaConf
from ..utils import initialize_hydra
from .base import BaseDataset


def load_dataset(
    dataset_name: str,
    config_path: Optional[str] = None,
) -> BaseDataset:
    """
    Download and instantiate a dataset using Hydra configuration.

    Args:
        dataset_name: Name of dataset as specified in config
        config_path: Optional path to config yaml file. If not provided,
                    will use only the package's default config.
    Returns:
        BaseDataset: Instantiated dataset object
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

    # Instantiate the dataset using Hydra
    dataset = instantiate(dataset_info)
    dataset.path = os.path.expanduser(dataset.path)

    return dataset


def list_available_datasets() -> List[str]:
    """
    Lists all available datasets defined in the datasets.yaml configuration file.

    Returns:
        list: A sorted list of dataset names available in the configuration.
    """
    initialize_hydra()

    # Load the datasets configuration
    cfg = OmegaConf.to_container(hydra.compose(config_name="datasets"), resolve=True)

    # Extract dataset names
    dataset_names = list(cfg.get("datasets", {}).keys())

    # Sort alphabetically for easier reading
    dataset_names.sort()

    return dataset_names