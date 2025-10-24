import os
from typing import Dict, Optional, Any
from pathlib import Path
from urllib.parse import urlparse
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from czbenchmarks.datasets.dataset import Dataset
from czbenchmarks.file_utils import download_file_from_remote
from czbenchmarks.utils import initialize_hydra, load_custom_config

logger = logging.getLogger(__name__)


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


def load_dataset(
    dataset_name: str,
) -> Dataset:
    """
    Load, download (if needed), and instantiate a dataset using Hydra configuration.

    Args:
        dataset_name (str): Name of the dataset as specified in the configuration.

    Returns:
        Dataset: Instantiated dataset object with data loaded.

    Raises:
        ValueError: If the specified dataset is not found in the configuration.

    Notes:
        - Uses Hydra for instantiation and configuration management.
        - Downloads dataset file if a remote path is specified using `download_file_from_remote`.
        - The returned dataset object is an instance of the `Dataset` class or its subclass.
    """
    initialize_hydra()
    cfg = hydra.compose(config_name="datasets")

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


def load_custom_dataset(
    dataset_name: str,
    custom_dataset_config_path: Optional[str] = None,
    custom_dataset_kwargs: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Instantiate a dataset with a custom configuration. This can include but
    is not limited to a local path for a custom dataset file and/or a
    dictionary of custom parameters to update the default configuration. If
    the dataset name does not exist in the default config, this function will
    add the dataset to the configuration.

    Args:
        dataset_name: The name of the dataset, either custom or from the config
        custom_dataset_config_path: Optional path to a YAML file containing a
            custom configuration that can be used to update the existing default configuration.
        custom_dataset_kwargs: Custom configuration dictionary to update the default
            configuration of the dataset class.
        cache_dir: Optional directory to cache the dataset file. If not provided,
            the global cache manager directory will be used.

    Returns:
        Instantiated dataset object with data loaded.

    Example:
        ```python
        from czbenchmarks.datasets.types import Organism
        from czbenchmarks.datasets.utils import load_custom_dataset

        custom_dataset_config_path = "/path/to/new_dataset.yaml"

        my_dataset_name = "my_dataset"
        custom_dataset_kwargs = {
            "organism": Organism.HUMAN,
            "path": "example-small.h5ad",
        }

        dataset = load_custom_dataset(
            dataset_name=my_dataset_name,
            custom_dataset_config_path=custom_dataset_config_path,
            custom_dataset_kwargs=custom_dataset_kwargs
        )
        ```
    """

    custom_cfg = load_custom_config(
        item_name=dataset_name,
        config_name="datasets",
        custom_config_path=custom_dataset_config_path,
        class_update_kwargs=custom_dataset_kwargs,
    )

    if "path" not in custom_cfg:
        raise ValueError(
            f"Path required but not found in resolved configuration: {custom_cfg}"
        )

    path = custom_cfg["path"]
    protocol = urlparse(str(path)).scheme

    if protocol:
        custom_cfg["path"] = download_file_from_remote(path, cache_dir=cache_dir)
    else:
        resolved_path = Path(path).expanduser().resolve()
        resolved_path = str(resolved_path)

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Local dataset file not found at path: {resolved_path}"
            )

        logger.info(f"Local dataset file found: {resolved_path}")
        custom_cfg["path"] = resolved_path

    dataset = instantiate(custom_cfg)
    dataset.load_data()

    return dataset
