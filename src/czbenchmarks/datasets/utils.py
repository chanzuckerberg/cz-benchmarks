import os
import hydra
from hydra.utils import instantiate
import boto3
import botocore
from botocore.config import Config
from typing import List, Optional
import yaml
from omegaconf import OmegaConf
from ..constants import DATASETS_CACHE_PATH
from ..utils import initialize_hydra
from .base import BaseDataset


def _download_dataset(uri: str, output_path: str, unsigned=True):
    """
    Download a dataset from the manifest file to the specified output path.

    Args:
        uri: S3 URI of dataset
        output_path: Local path where dataset should be downloaded
        unsigned: Whether to use unsigned requests (default: True)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Parse S3 URL
    bucket = uri.split("/")[2]
    key = "/".join(uri.split("/")[3:])

    # Download from S3
    s3_client = boto3.client(
        "s3",
        config=Config(signature_version=botocore.UNSIGNED) if unsigned else None,
    )
    s3_client.download_file(bucket, key, output_path)


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


_DATASET_TO_DISPLAY_NAME = {
    "adamson_perturb": "Adamson",
    "norman_perturb": "Norman",
    "dixit_perturb": "Dixit",
    "replogle_k562_perturb": "Replogle K562",
    "replogle_rpe1_perturb": "Replogle RPE1",
    "human_spermatogenesis": "Spermatogenesis - Homo sapiens",
    "mouse_spermatogenesis": "Spermatogenesis - Mus musculus",
    "rhesus_macaque_spermatogenesis": "Spermatogenesis - Macaca mulatta",
    "gorilla_spermatogenesis": "Spermatogenesis - Gorilla gorilla",
    "chimpanzee_spermatogenesis": "Spermatogenesis - Pan troglodytes",
    "marmoset_spermatogenesis": "Spermatogenesis - Callithrix jacchus",
    "chicken_spermatogenesis": "Spermatogenesis - Gallus gallus",
    "opossum_spermatogenesis": "Spermatogenesis - Monodelphis domestica",
    "platypus_spermatogenesis": "Spermatogenesis - Ornithorhynchus anatinus",
    "tsv2_bladder": "Tabula Sapiens 2.0 - Bladder",
    "tsv2_blood": "Tabula Sapiens 2.0 - Blood",
    "tsv2_bone_marrow": "Tabula Sapiens 2.0 - Bone marrow",
    "tsv2_ear": "Tabula Sapiens 2.0 - Ear",
    "tsv2_eye": "Tabula Sapiens 2.0 - Eye",
    "tsv2_fat": "Tabula Sapiens 2.0 - Fat",
    "tsv2_heart": "Tabula Sapiens 2.0 - Heart",
    "tsv2_large_intestine": "Tabula Sapiens 2.0 - Large intestine",
    "tsv2_liver": "Tabula Sapiens 2.0 - Liver",
    "tsv2_lung": "Tabula Sapiens 2.0 - Lung",
    "tsv2_lymph_node": "Tabula Sapiens 2.0 - Lymph node",
    "tsv2_mammary": "Tabula Sapiens 2.0 - Mammary",
    "tsv2_muscle": "Tabula Sapiens 2.0 - Muscle",
    "tsv2_ovary": "Tabula Sapiens 2.0 - Ovary",
    "tsv2_prostate": "Tabula Sapiens 2.0 - Prostate",
    "tsv2_salivary_gland": "Tabula Sapiens 2.0 - Salivary gland",
    "tsv2_skin": "Tabula Sapiens 2.0 - Skin",
    "tsv2_small_intestine": "Tabula Sapiens 2.0 - Small intestine",
    "tsv2_spleen": "Tabula Sapiens 2.0 - Spleen",
    "tsv2_stomach": "Tabula Sapiens 2.0 - Stomach",
    "tsv2_testis": "Tabula Sapiens 2.0 - Testis",
    "tsv2_thymus": "Tabula Sapiens 2.0 - Thymus",
    "tsv2_tongue": "Tabula Sapiens 2.0 - Tongue",
    "tsv2_trachea": "Tabula Sapiens 2.0 - Trachea",
    "tsv2_uterus": "Tabula Sapiens 2.0 - Uterus",
    "tsv2_vasculature": "Tabula Sapiens 2.0 - Vasculature",
}


def dataset_to_display_name(dataset_name: str) -> str:
    """try to map dataset names to more uniform, pretty strings"""
    try:
        return _DATASET_TO_DISPLAY_NAME[dataset_name]
    except KeyError:
        # e.g. "my_awesome_dataset" -> "My awesome dataset"
        parts = dataset_name.split("_")
        return " ".join((parts[0].title(), *(part.lower() for part in parts[1:])))
