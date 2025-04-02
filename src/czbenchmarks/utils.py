import os
from datetime import datetime

import boto3
from botocore.config import Config
import botocore
import logging
import hydra
from omegaconf import OmegaConf


logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("botocore.httpchecksum").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def initialize_hydra(config_path="./conf"):
    """
    Initialize Hydra configuration system.

    Args:
        config_path: Path to the configuration directory
    """
    if hydra.core.global_hydra.GlobalHydra.instance().is_initialized():
        hydra.core.global_hydra.GlobalHydra.instance().clear()

    hydra.initialize(
        config_path=config_path,
        version_base=None,
    )


def import_class_from_config(config_path: str):
    """
    Import a class based on the _target_ field in a configuration file.

    Args:
        config_path: Path to the configuration file

    Returns:
        class_obj: The imported class object
    """
    # Load the configuration
    logger.info(f"Loading model configuration from {config_path}")
    cfg = OmegaConf.load(config_path)

    # Get the target class path
    target_path = cfg._target_

    # Import the class using the target path
    module_path, class_name = target_path.rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    class_obj = getattr(module, class_name)

    logger.info(f"Imported class: {class_obj.__name__}")

    return class_obj


def download_s3_file(bucket, key, local_path, unsigned=True):
    """
    Downloads a single file from S3 to a local path.

    :param bucket: S3 bucket name
    :param key: S3 key (file path) to download
    :param local_path: Local file path to save to
    :param unsigned: Whether to use unsigned requests (default: True)
    """
    s3 = boto3.client(
        "s3",
        config=Config(signature_version=botocore.UNSIGNED) if unsigned else None,
    )

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {key} from s3://{bucket} to {local_path}")
    except Exception as e:
        logger.error(f"Failed to download {key} from s3://{bucket}: {str(e)}")
        raise


def sync_s3_to_local(bucket, prefix, local_dir, unsigned=True):
    """
    Syncs files from an S3 bucket prefix to a local directory.

    :param bucket: S3 bucket name
    :param prefix: S3 prefix (directory path or file) to sync from
    :param local_dir: Local directory path to sync to
    :param unsigned: Whether to use unsigned requests (default: True)
    """
    s3 = boto3.client(
        "s3",
        config=Config(signature_version=botocore.UNSIGNED) if unsigned else None,
    )

    # Prefix is a directory, proceed with original logic
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            key = obj["Key"]
            # Skip keys that don't start with the prefix (unlikely due to Paginator)
            if not key.startswith(prefix):
                continue
            # Skip directory markers
            if key.endswith("/"):
                continue
            # Calculate relative path and local file path
            relative_key = key[len(prefix) :].lstrip("/")
            if not relative_key:
                continue  # Skip the exact prefix if it's an object
            local_file_path = os.path.join(local_dir, relative_key)
            directory = os.path.dirname(local_file_path)
            if directory:  # Ensure the directory exists
                os.makedirs(directory, exist_ok=True)

            # Check if download/update is needed
            if not os.path.exists(local_file_path):
                s3.download_file(bucket, key, local_file_path)
                logger.info(f"Downloaded: {relative_key} to {local_file_path}")
            else:
                # Compare last modified times
                s3_time = obj["LastModified"].replace(tzinfo=None)
                local_time = datetime.utcfromtimestamp(
                    os.path.getmtime(local_file_path)
                )
                if s3_time > local_time:
                    s3.download_file(bucket, key, local_file_path)
                    logger.info(f"Updated: {relative_key} at {local_file_path}")
