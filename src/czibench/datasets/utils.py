import os
import hydra
from hydra.utils import instantiate
import boto3
from typing import Optional
import yaml
from omegaconf import OmegaConf
from ..constants import DATASETS_CACHE_PATH
from .base import BaseDataset
import re


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
    if uri.startswith("s3://"):
        bucket = uri.split("/")[2]
        key = "/".join(uri.split("/")[3:])

        # Download from S3
        s3_client = boto3.client("s3")
        s3_client.download_file(bucket, key, output_path)
    elif uri.startswith("http://") or uri.startswith("https://"):
        # Download from HTTP(S)
        import requests

        response = requests.get(uri)
        with open(output_path, "wb") as f:
            f.write(response.content)
    else:
        print(f"Unsupported URI scheme: {uri}")
        return


def init_dataset(
    dataset_name: str,
    config_path: Optional[str] = None,
) -> BaseDataset:
    """
    Instantiate a dataset using Hydra configuration.

    Args:
        dataset_name: Name of dataset as specified in config
        config_path: Optional path to config yaml file. If not provided,
                    will use only the package's default config.
    Returns:
        BaseDataset: Instantiated dataset object
    """
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    # Initialize Hydra with package config
    hydra.initialize(
        config_path="../../../conf",
        version_base=None,
    )

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
    dataset.source_path = os.path.expanduser(dataset_info.source_path)
    
    # TODO: For remote model inference, single cell benchmarks only need the 
    # `obs` dataframe, so it is a bit excessive to download the entire H5AD (`X` data).
    _fetch_dataset_input_data(dataset)

    return dataset


def _fetch_dataset_input_data(dataset: BaseDataset):
    """
    Download the input dataset and update dataset.local_path to point at local file

    Args:
        dataset: The dataset object
    """
    is_remote_path = re.match(r".*?://", dataset.source_path) is not None

    if is_remote_path:
        # Setup cache path
        cache_path = os.path.expanduser(DATASETS_CACHE_PATH)
        os.makedirs(cache_path, exist_ok=True)
        # Remove URL parameters if present
        source_path_clean = dataset.source_path.split('?')[0]
        cache_file = os.path.join(cache_path, os.path.basename(source_path_clean))

        # Only download if file doesn't exist
        if not os.path.exists(cache_file):
            _download_dataset(dataset.source_path, cache_file)

        # Update path to cached file
        dataset.local_path = cache_file
    else:
        dataset.local_path = os.path.expanduser(dataset.source_path)
        if not os.path.exists(dataset.local_path):
            raise FileNotFoundError(f"Local dataset file not found: {dataset.local_path}")
        
