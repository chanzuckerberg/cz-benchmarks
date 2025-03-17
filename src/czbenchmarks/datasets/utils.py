import os
import hydra
from hydra.utils import instantiate
import boto3
from typing import List, Optional
import yaml
from omegaconf import OmegaConf
from ..constants import DATASETS_CACHE_PATH
from .base import BaseDataset


def _download_dataset(uri: str, output_path: str):
    """
    Download a dataset from the manifest file to the specified output path.

    Args:
        dataset_name: Name of dataset as specified in manifest
        output_path: Local path where dataset should be downloaded
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Parse S3 URL
    bucket = uri.split("/")[2]
    key = "/".join(uri.split("/")[3:])

    # Download from S3
    s3_client = boto3.client("s3")
    s3_client.download_file(bucket, key, output_path)


def initialize_hydra(config_path="../../../conf"):
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    hydra.initialize(
        config_path=config_path,
        version_base=None,
    )


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
    original_path = dataset_info.path

    is_s3_path = original_path.startswith("s3://")
    expanded_path = os.path.expanduser(original_path)

    if not is_s3_path:
        if not os.path.exists(expanded_path):
            raise FileNotFoundError(f"Local dataset file not found: {expanded_path}")
    else:
        # Setup cache path
        cache_path = os.path.expanduser(DATASETS_CACHE_PATH)
        os.makedirs(cache_path, exist_ok=True)
        cache_file = os.path.join(cache_path, f"{dataset_name}.h5ad")

        # Only download if file doesn't exist
        if not os.path.exists(cache_file):
            _download_dataset(original_path, cache_file)

        # Update path to cached file
        dataset_info.path = cache_file

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
