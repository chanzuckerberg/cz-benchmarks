from datetime import datetime
import mimetypes
import os
from pathlib import Path
import logging

import boto3
from botocore.config import Config
import botocore
import hydra
from mypy_boto3_s3.client import S3Client
from omegaconf import OmegaConf

from .exceptions import RemoteStorageError, RemoteStorageObjectAlreadyExists


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


def _get_s3_client(make_unsigned_request: bool = False) -> S3Client:
    if make_unsigned_request:
        return boto3.client("s3", config=Config(signature_version=botocore.UNSIGNED))
    else:
        return boto3.client("s3")


def _does_remote_exist(s3_client: S3Client, bucket: str, key: str):
    """return if the key exists in the remote S3 bucket"""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return False
    except botocore.exceptions.BotoCoreError as e:
        raise RemoteStorageError(
            f"Error checking for existence of 's3://{bucket}/{key}'"
        ) from e
    return True


def upload_file_to_remote(
    local_file: str | Path,
    remote_prefix_url: str,
    make_unsigned_request: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """Upload a local file to an s3 prefix, preserving the filename remotely"""
    local_file = Path(local_file)
    if not local_file.is_file():
        raise FileNotFoundError(f"{local_file!r} does not exist")
    filename = local_file.name

    if not remote_prefix_url.endswith("/"):
        raise ValueError(
            f"Remote URL {remote_prefix_url!r} should be a prefix ending in '/'"
        )
    else:
        bucket, key_prefix = remote_prefix_url.removeprefix("s3://").split("/", 1)

    remote_key = f"{key_prefix.rstrip('/')}/{filename}"
    s3 = _get_s3_client(make_unsigned_request)

    if not overwrite_existing:
        if _does_remote_exist(bucket, remote_key):
            raise RemoteStorageObjectAlreadyExists(
                f"Remote file already exists at 's3://{bucket}/{remote_key}'"
            )
    try:
        s3.upload_file(str(local_file), bucket, remote_key)
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        raise RemoteStorageError(
            f"Failed to upload {local_file!r} to 's3://{bucket}/{remote_key}'"
        ) from e


def upload_blob_to_remote(
    blob: bytes,
    remote_url: str,
    make_unsigned_request: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """Upload the contents of a text buffer to the exact S3 location given by remote_url."""
    try:
        bucket, remote_key = remote_url.removeprefix("s3://").split("/", 1)
    except ValueError:
        raise ValueError(
            f"Remote URL {remote_url!r} is missing a key to a specific object"
        )

    s3 = _get_s3_client(make_unsigned_request)
    if not overwrite_existing:
        if _does_remote_exist(bucket, remote_key):
            raise RemoteStorageObjectAlreadyExists(
                f"Remote file already exists at 's3://{bucket}/{remote_key}'"
            )
    try:
        content_type, _ = mimetypes.guess_type(remote_key)
        s3.put_object(
            Bucket=bucket,
            Key=remote_key,
            Body=blob,
            ContentType=content_type or "application/octet-stream",
        )
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        raise RemoteStorageError(
            f"Failed to upload to 's3://{bucket}/{remote_key}'"
        ) from e


def download_file_from_remote(
    remote_url: str,
    local_directory: str | Path,
    local_filename: str | None = None,
    make_unsigned_request: bool = True,
) -> None:
    """Download a remote file from s3 to a local directory, preserving the filename if not specified"""
    try:
        bucket, remote_key = remote_url.removeprefix("s3://").split("/", 1)
    except ValueError:
        raise ValueError(
            f"Remote URL {remote_url!r} is missing a key to a specific object"
        )

    local_directory = Path(local_directory)
    local_directory.mkdir(parents=True, exist_ok=True)
    if local_filename is None:
        _, local_filename = remote_url.rsplit("/", 1)

    local_file = local_directory / local_filename
    s3 = _get_s3_client(make_unsigned_request)
    try:
        s3.download_file(bucket, remote_key, str(local_file))
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        raise RemoteStorageError(
            f"Failed to download 's3://{bucket}/{remote_key}' to {local_file!r}"
        ) from e
