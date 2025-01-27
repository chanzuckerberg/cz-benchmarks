import os
import yaml
import boto3
from ..constants import DATASETS_CACHE_PATH


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
    bucket = uri.split('/')[2]
    key = '/'.join(uri.split('/')[3:])
    
    # Download from S3
    s3_client = boto3.client('s3')
    s3_client.download_file(bucket, key, output_path)

def download_dataset(
    dataset_name: str
) -> str:
    # Load manifest
    manifest_path = os.path.join(os.path.dirname(__file__), "dataset_manifest.yaml")
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)["datasets"]

    cache_path = os.path.expanduser(DATASETS_CACHE_PATH)
    os.makedirs(cache_path, exist_ok=True)
    
    # Handle manifest dataset
    if dataset_name not in manifest:
        raise ValueError(f"Dataset {dataset_name} not found in manifest")
        
    dataset_info = manifest[dataset_name]
    
    os.makedirs(cache_path, exist_ok=True)
    cache_file = os.path.join(cache_path, f"{dataset_name}.h5ad")
    _download_dataset(dataset_info["path"], cache_file)
    
    return cache_file