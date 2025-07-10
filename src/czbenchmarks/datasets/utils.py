import os
import hydra
from hydra.utils import instantiate
from typing import List, Optional
import yaml
from omegaconf import OmegaConf
from ..utils import initialize_hydra
from .base import BaseDataset
from ..constants import DATASETS_CACHE_PATH
import boto3
from pathlib import Path
import logging
from tqdm import tqdm

log = logging.getLogger(__name__)

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

    # Handle local caching and remote downloading
    dataset_path = _handle_dataset_path(dataset_info["path"])

    # Instantiate the dataset using Hydra
    dataset = instantiate(dataset_info)
    dataset.path = dataset_path #os.path.expanduser(dataset.path)

    # Load the dataset into memory
    dataset.load_data()

    return dataset

def _handle_dataset_path(remote_path: str) -> str:
    """
    Handle dataset path by checking local cache or downloading from remote sources.

    Args:
        remote_path: Remote path (e.g., S3 URL) of the dataset.

    Returns:
        str: Local path to the dataset file.
    """
    cache_dir = Path(DATASETS_CACHE_PATH).expanduser()
    local_path = cache_dir / Path(remote_path).name

    # Check if the dataset is already cached locally
    if local_path.exists():
        log.info(f"Dataset available in local cache: {local_path}")
        return str(local_path)

    # Download the dataset if it's not cached
    if remote_path.startswith("s3://"):
        log.info(f"Downloading dataset from S3: {remote_path} to {local_path}")
        _download_from_s3(remote_path, local_path)
    else:
        raise ValueError(f"Unsupported remote path: {remote_path}")

    return str(local_path)


def _download_from_s3(remote_path: str, local_path: Path) -> None:
    """
    Download a dataset from an S3 bucket.

    Args:
        remote_path: S3 path of the dataset.
        local_path: Local path to save the downloaded dataset.
    """
    s3 = boto3.client("s3")
    bucket, key = _parse_s3_path(remote_path)

    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the size of the file for progress bar
        obj = s3.head_object(Bucket=bucket, Key=key)
        total_size = obj.get('ContentLength', 0)

        with tqdm(total=total_size, unit='B', 
                  unit_scale=True, 
                  desc=f"Downloading {local_path.name}") as pbar:
            
            def progress_hook(bytes_amount):
                pbar.update(bytes_amount)

            s3.download_file(
            bucket,
            key,
            str(local_path),
            Callback=progress_hook
            )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download dataset from S3: {remote_path}. "
            f"Error: {str(e)}"
        ) from e


def _parse_s3_path(s3_path: str) -> tuple[str, str]:
    """
    Parse an S3 path into bucket and key.

    Args:
        s3_path: S3 path (e.g., "s3://bucket/key").

    Returns:
        tuple[str, str]: Bucket name and key.
    """
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}")
    parts = s3_path[5:].split("/", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid S3 path: {s3_path}")
    return parts[0], parts[1]


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
