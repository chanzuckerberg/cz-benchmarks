import os
from pathlib import Path
from datetime import datetime, timedelta
import boto3
from botocore.config import Config
import botocore
import logging
from czbenchmarks.constants import DATASETS_CACHE_PATH
from czbenchmarks.exceptions import RemoteStorageError, RemoteStorageObjectAlreadyExists
import mimetypes

log = logging.getLogger(__name__)

# Global cache manager instance
DEFAULT_CACHE_DIR = os.getenv("DATASETS_CACHE_PATH", DATASETS_CACHE_PATH)
DEFAULT_CACHE_EXPIRATION_DAYS = int(os.getenv("CZBENCHMARKS_CACHE_EXPIRATION_DAYS", 30))


class CacheManager:
    """Centralized cache management for remote files."""

    def __init__(
        self,
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        expiration_days: int = DEFAULT_CACHE_EXPIRATION_DAYS,
    ):
        self.cache_dir = Path(cache_dir).expanduser()
        self.expiration_days = expiration_days
        self.ensure_directory_exists(self.cache_dir)

    def ensure_directory_exists(self, directory: Path) -> None:
        """Ensure the given directory exists."""
        directory.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, remote_url: str) -> Path:
        """Generate a local cache path for a remote file."""
        filename = Path(remote_url).name
        return self.cache_dir / filename

    def is_expired(self, file_path: Path) -> bool:
        """Check if a cached file is expired."""
        if not file_path.exists():
            return True
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        return datetime.now() - last_modified > timedelta(days=self.expiration_days)

    def clean_expired_cache(self) -> None:
        """Clean up expired cache files."""
        for file in self.cache_dir.iterdir():
            if self.is_expired(file):
                log.info(f"Removing expired cache file: {file}")
                file.unlink()


# Default cache manager instance
_default_cache_manager = CacheManager()


def _get_s3_client(make_unsigned_request: bool = True) -> boto3.client:
    """Get an S3 client with optional unsigned requests."""
    if make_unsigned_request:
        return boto3.client("s3", config=Config(signature_version=botocore.UNSIGNED))
    else:
        return boto3.client("s3")


def _get_remote_last_modified(
    s3_client: boto3.client, bucket: str, key: str
) -> datetime | None:
    """Return the LastModified timestamp of the remote S3 object, or None if it doesn't exist."""
    try:
        resp = s3_client.head_object(Bucket=bucket, Key=key)
        return resp["LastModified"]
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] == "404":
            return None
        raise RemoteStorageError(
            f"Error checking existence of 's3://{bucket}/{key}'"
        ) from e
    except botocore.exceptions.BotoCoreError as e:
        raise RemoteStorageError(
            f"Error checking existence of 's3://{bucket}/{key}'"
        ) from e


def download_file_from_remote(
    remote_url: str,
    cache_dir: str | Path = None,
    make_unsigned_request: bool = True,
    validate_integrity: bool = False,
) -> str:
    """
    Download a remote file to a local cache directory.

    Args:
        remote_url (str): Remote URL of the file (e.g., S3 path).
        cache_dir (str | Path, optional): Local directory to save the file. Defaults to the global cache manager's directory.
        make_unsigned_request (bool, optional): Whether to use unsigned requests for S3 (default: True).
        validate_integrity (bool, optional): Whether to validate file integrity using checksums (default: False).

    Returns:
        str: Local path to the downloaded file.

    Raises:
        ValueError: If the remote URL is invalid.
        RemoteStorageError: If the file download fails due to S3 errors.

    Notes:
        - If the file already exists in the cache and is not expired, it will not be downloaded again.
        - Unsigned requests are tried first, followed by signed requests if the former fails.
    """
    cache_manager = (
        _default_cache_manager if cache_dir is None else CacheManager(cache_dir)
    )
    try:
        bucket, remote_key = remote_url.removeprefix("s3://").split("/", 1)
    except ValueError:
        raise ValueError(f"Invalid remote URL: {remote_url}")

    local_file = cache_manager.get_cache_path(remote_url)
    if local_file.exists() and not cache_manager.is_expired(local_file):
        log.info(f"File already exists in cache: {local_file}")
        return str(local_file)

    s3 = _get_s3_client(make_unsigned_request)
    try:
        s3.download_file(bucket, remote_key, str(local_file))
    except botocore.exceptions.ClientError:
        if not make_unsigned_request:
            raise
        log.warning("Unsigned request failed. Trying signed request.")
        s3 = _get_s3_client(make_unsigned_request=False)
        s3.download_file(bucket, remote_key, str(local_file))
    except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
        raise RemoteStorageError(
            f"Failed to download {remote_url} to {local_file}"
        ) from e

    log.info(f"Downloaded file to cache: {local_file}")
    return str(local_file)


def upload_file_to_remote(
    local_file: str | Path,
    remote_prefix_url: str,
    make_unsigned_request: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """
    Upload a local file to an S3 prefix, preserving the filename remotely.

    Args:
        local_file (str | Path): Path to the local file to upload.
        remote_prefix_url (str): S3 prefix URL where the file will be uploaded (must end with '/').
        make_unsigned_request (bool, optional): Whether to use unsigned requests for S3 (default: False).
        overwrite_existing (bool, optional): Whether to overwrite the file if it already exists remotely (default: False).

    Raises:
        FileNotFoundError: If the local file does not exist.
        ValueError: If the remote prefix URL is invalid.
        RemoteStorageObjectAlreadyExists: If the file exists remotely and overwrite_existing is False.
        RemoteStorageError: If the upload fails due to S3 errors.

    Notes:
        - The filename of the local file is preserved in the remote location.
        - If overwrite_existing is False, the function checks for the existence of the remote file before uploading.
    """
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
        if _get_remote_last_modified(s3, bucket, remote_key) is not None:
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
    """
    Upload the contents of a bytes buffer to the exact S3 location given by remote_url.

    Args:
        blob (bytes): Bytes to upload.
        remote_url (str): S3 URL of the destination object (e.g., 's3://bucket/key').
        make_unsigned_request (bool, optional): Whether to use unsigned requests for S3 (default: False).
        overwrite_existing (bool, optional): Whether to overwrite the object if it already exists remotely (default: False).

    Raises:
        ValueError: If the remote URL is invalid.
        RemoteStorageObjectAlreadyExists: If the object exists remotely and overwrite_existing is False.
        RemoteStorageError: If the upload fails due to S3 errors.

    Notes:
        - The function determines the MIME type of the object based on its key and sets the ContentType accordingly.
        - If overwrite_existing is False, the function checks for the existence of the remote object before uploading.
    """

    try:
        bucket, remote_key = remote_url.removeprefix("s3://").split("/", 1)
    except ValueError:
        raise ValueError(
            f"Remote URL {remote_url!r} is missing a key to a specific object"
        )

    s3 = _get_s3_client(make_unsigned_request)
    if not overwrite_existing:
        if _get_remote_last_modified(s3, bucket, remote_key) is not None:
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
    log.info(f"Uploaded blob to remote: s3://{bucket}/{remote_key}")
