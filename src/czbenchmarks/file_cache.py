"""
File caching utilities for remote storage operations.

This module provides functions for caching datasets and model outputs to/from remote storage,
primarily AWS S3. It includes functionality for downloading, uploading, and managing cached
processed datasets.
"""

import argparse
import logging

from pydantic import BaseModel


log = logging.getLogger(__name__)


class CacheOptions(BaseModel):
    """
    Configuration options for caching datasets and model outputs.

    Attributes:
        download_embeddings (bool): Whether to download embeddings from the remote cache.
        upload_embeddings (bool): Whether to upload embeddings to the remote cache.
        upload_results (bool): Whether to upload results to the remote cache.
        remote_cache_url (str): URL of the remote cache.
    """

    download_embeddings: bool
    upload_embeddings: bool
    upload_results: bool
    remote_cache_url: str

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "CacheOptions":
        remote_cache_url = args.remote_cache_url or ""
        return cls(
            remote_cache_url=remote_cache_url,
            download_embeddings=bool(remote_cache_url)
            and args.remote_cache_download_embeddings,
            upload_embeddings=bool(remote_cache_url)
            and args.remote_cache_upload_embeddings,
            upload_results=bool(remote_cache_url) and args.remote_cache_upload_results,
        )
