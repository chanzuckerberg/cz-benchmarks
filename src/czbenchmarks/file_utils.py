import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import boto3
import botocore
from botocore.config import Config

from czbenchmarks.constants import DATASETS_CACHE_PATH
from czbenchmarks.exceptions import RemoteStorageError

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


def download_file_from_remote(
    remote_url: str,
    cache_dir: str | Path = None,
    make_unsigned_request: bool = True,
) -> str:
    """
    Download a remote file to a local cache directory.

    Args:
        remote_url (str): Remote URL of the file (e.g., S3 path).
        cache_dir (str | Path, optional): Local directory to save the file. Defaults to the global cache manager's directory.
        make_unsigned_request (bool, optional): Whether to use unsigned requests for S3 (default: True).

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

    protocol = urlparse(remote_url).scheme
    if not protocol:
        raise ValueError(f"Only S3 paths are supported, got local path: {remote_url}")
    elif protocol != "s3":
        raise ValueError(
            f"Unsupported protocol {protocol} for remote URL: {remote_url}"
        )
    else:
        bucket, remote_key = remote_url.removeprefix("s3://").split("/", 1)

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
