import os
import hydra
from hydra.utils import instantiate
from typing import Dict, Optional
import yaml
from omegaconf import OmegaConf
from czbenchmarks.datasets.types import Organism
from .dataset import Dataset
from czbenchmarks.utils import initialize_hydra
from czbenchmarks.file_utils import download_file_from_remote
import logging

log = logging.getLogger(__name__)


def _generate_dataset_name_from_path(path: str) -> str:
    """
    Generate a unique dataset name from a file path.
    
    Args:
        path (str): File path to convert to dataset name
        
    Returns:
        str: A valid dataset name derived from the path
    """
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(path))[0]
    # Replace invalid characters with underscores
    safe_name = "".join(c if c.isalnum() else "_" for c in filename)
    # Remove consecutive underscores and strip trailing ones
    safe_name = "_".join(filter(None, safe_name.split("_")))
    return f"user_dataset_{safe_name}"


def _handle_dataset_path(path: str) -> str:
    """
    Handle dataset path for both local files (file://) and remote files (s3://).
    
    Args:
        path (str): Dataset path, either local (file://) or remote (s3://)
        
    Returns:
        str: Local file path to the dataset
        
    Raises:
        FileNotFoundError: If local file doesn't exist
        ValueError: If file:// path is invalid
    """
    if path.startswith("file://"):
        # Handle local file
        local_path = path[7:]  # Remove "file://" prefix
        
        # Convert to absolute path and expand user directory
        local_path = os.path.expanduser(local_path)
        local_path = os.path.abspath(local_path)
        
        # Check if file exists
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local dataset file not found: {local_path}")
        
        log.info(f"Using local dataset file: {local_path}")
        return local_path
    else:
        # Handle remote file (existing functionality)
        return download_file_from_remote(path)


def load_local_dataset(
    _target_: str,
    organism: str,
    path: str,
    **kwargs,
) -> Dataset:
    """
    Instantiate a dataset directly from arguments without requiring a YAML file.

    This is a convenience wrapper that builds a configuration on the fly and
    passes it to the core `load_dataset` function.

    Args:
        _target_ (str): The full import path to the Dataset class to instantiate.
        organism (str): The organism of the dataset.
        path (str): The local or remote path to the dataset file.
        **kwargs: Additional key-value pairs for the dataset config.

    Returns:
        Dataset: Instantiated dataset object with data loaded.

    Example:
        dataset = load_local_dataset(
            _target_="czbenchmarks.datasets.SingleCellLabeledDataset",
            organism="HUMAN",
            path="example-small.h5ad",
        )
    """

    # Validate _target_
    if not isinstance(_target_, str) or not _target_:
        raise ValueError("The '_target_' argument must be a non-empty string specifying the full import path of the dataset class (e.g., 'czbenchmarks.datasets.SingleCellLabeledDataset').")
    if not _target_.startswith("czbenchmarks.datasets."):
        raise ValueError(f"The provided '_target_' value '{_target_}' is invalid. It must start with 'czbenchmarks.datasets.' and refer to a valid dataset class import path.")

    # Validate organism
    if not isinstance(organism, str) or not organism:
        raise ValueError(f"You must provide a non-empty string for 'organism'. It should be one of: {list(Organism.__members__.keys())}")

    if organism.upper() not in Organism.__members__:
        raise ValueError(f"Unknown organism: {organism}. Must be one of: {list(Organism.__members__.keys())}")

    # Validate path
    if not isinstance(path, str) or not path:
        raise ValueError("You must provide a non-empty string for 'path'.")
        
    # Generate a unique dataset name from the file path
    dataset_name = _generate_dataset_name_from_path(path)

    # Combine all config arguments into a single dictionary
    config_args = {"_target_": _target_, "path": path, "organism": organism, **kwargs}
    
    # Automatically prepend 'file://' to local paths if no scheme is present
    if "://" not in config_args["path"]:
        config_args["path"] = f"file://{config_args['path']}"

    # Build the OmegaConf object in the required structure
    config = OmegaConf.create({"datasets": {dataset_name: config_args}})

    # Call the core loader with the generated config
    return load_dataset(dataset_name=dataset_name, config=config)


def load_dataset(
    dataset_name: str,
    config_path: Optional[str] = None,
    config: Optional[OmegaConf] = None,
) -> Dataset:
    """
    Load, download (if needed), and instantiate a dataset using Hydra configuration.

    This function serves as the core loader, using a configuration sourced from
    the default location, a YAML file, or a directly provided OmegaConf object.

    Args:
        dataset_name (str): Name of the dataset as specified in the configuration.
        config_path (Optional[str]): Optional path to a custom config YAML file.
        config (Optional[OmegaConf]): An optional, pre-loaded OmegaConf object to use.
            If provided, `config_path` and default configs are ignored.

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
    if config is not None:
        # Use the provided config directly
        cfg = config
    else:
        # Load configuration from files
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
        raise ValueError(f"Dataset '{dataset_name}' not found in the provided configuration")

    dataset_info = cfg.datasets[dataset_name]

    # Handle local caching and remote downloading
    dataset_info["path"] = _handle_dataset_path(dataset_info["path"])

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
