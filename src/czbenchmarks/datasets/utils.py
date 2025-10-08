import os
from typing import Dict, Optional, Any, Dict
from pathlib import Path
from urllib.parse import urlparse
import logging

import hydra
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from czbenchmarks.datasets.dataset import Dataset
from czbenchmarks.file_utils import download_file_from_remote
from czbenchmarks.utils import initialize_hydra, load_custom_config

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str,
    config_path: Optional[str] = None,
) -> Dataset:
    """
    Load, download (if needed), and instantiate a dataset using Hydra configuration.

    Args:
        dataset_name (str): Name of the dataset as specified in the configuration.
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
    dataset.load_data()

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


def load_customized_dataset(
    dataset_name: str,
    custom_dataset_config: Dict[str, Any],
) -> Dataset:
    """
    Instantiate a dataset with a custom configuration. This can include but
    is not limited to a local path for a custom dataset file. If the dataset
    exists in the default config, this function will update parameters.
    Otherwise, it will add the dataset to the default config.

    This function is completely independent from load_dataset() and directly
    instantiates the dataset class without using OmegaConf objects.

    Args:
        dataset_name: The name of the dataset, either custom or from the config
        custom_class_init: Custom configuration dictionary for the dataset.

    Returns:
        Instantiated dataset object with data loaded.

    Example:
        ```python
        from czbenchmarks.datasets.types import Organism
        from czbenchmarks.datasets.utils import load_customized_dataset

        my_dataset_name = "my_dataset"
        custom_dataset_config = {
            "organism": Organism.HUMAN,
            "path": "example-small.h5ad",
        }
        dataset = load_customized_dataset(
            dataset_name=my_dataset_name,
            custom_dataset_config=custom_dataset_config
        )
        ```
    """

    custom_cfg = load_custom_config(
        item_name=dataset_name,
        config_name="datasets",
        class_update_kwargs=custom_dataset_config,
    )

    if "path" not in custom_cfg:
        raise ValueError(
            f"Path required but not found in harmonized configuration: {custom_cfg}"
        )

    path = custom_cfg.pop("path")
    protocol = urlparse(str(path)).scheme

    if protocol:
        # download_file_from_remote expects an s3 path
        if protocol == "s3":
            custom_cfg["path"] = download_file_from_remote(path)
        else:
            raise ValueError(
                f"Remote protocols other than s3 are not currently supported: {path}"
            )
    else:
        logger.info(f"Local dataset file found: {path}")
        resolved_path = Path(path).expanduser().resolve()

        if not resolved_path.exists():
            raise FileNotFoundError(f"Local dataset file not found: {resolved_path}")

        custom_cfg["path"] = str(resolved_path)

    dataset = instantiate(custom_cfg)
    dataset.load_data()

    return dataset
