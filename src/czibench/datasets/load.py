from typing import Type, Optional, Union
import os
import tempfile
import yaml
import boto3
from .base import BaseDataset
from .sc import SingleCellDataset
from .types import Organism


def download_dataset(uri: str, output_path: str):
    """
    Download a dataset from the manifest file to the specified output path.
    
    Args:
        dataset_name: Name of dataset as specified in manifest
        output_path: Local path where dataset should be downloaded
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Parse S3 URL
    bucket = uri.split('/')[2]
    key = '/'.join(uri.split('/')[3:])
    
    # Download from S3
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, key, output_path)

def load_dataset(
    dataset_name_or_path: str,
    dataset_type: Optional[Type[BaseDataset]] = None,
    organism: Optional[Union[str, Organism]] = None
) -> BaseDataset:
    """
    Load a dataset using the specified dataset type.
    
    Args:
        dataset_name_or_path: Name from manifest or path to local file
        dataset_type: Optional dataset type class (required for local files)
        organism: Optional organism specification (required for local files)
        
    Returns:
        BaseDataset: Loaded dataset using appropriate container
    """
    # Load manifest
    manifest_path = os.path.join(os.path.dirname(__file__), "dataset_manifest.yaml")
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)["datasets"]

    # Handle local file
    if dataset_name_or_path.endswith('.h5ad') and os.path.exists(dataset_name_or_path):
        if dataset_type is None:
            raise ValueError("dataset_type must be specified for local files")
        if organism is None and dataset_type is SingleCellDataset:
            raise ValueError("organism must be specified for local files")
            
        if isinstance(organism, str):
            organism = Organism[organism.upper()]
            
        return dataset_type(
            path=dataset_name_or_path,
            organism=organism
        )

    # Handle manifest dataset
    if dataset_name_or_path not in manifest:
        raise ValueError(f"Dataset {dataset_name_or_path} not found in manifest")
        
    dataset_info = manifest[dataset_name_or_path]
    
    # Get container class from string name
    container_name = dataset_info["dataset_type"]
    container_class = globals()[container_name]
    
    # Download to temp location
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = os.path.join(temp_dir, f"{dataset_name_or_path}.h5ad")
        download_dataset(dataset_info["path"], temp_path)
        
        return container_class(
            path=temp_path,
            organism=Organism[dataset_info["organism"].upper()],
            **dataset_info.get("metadata", {})
        )
