"""
File caching utilities for remote storage operations.

This module provides functions for caching datasets and model outputs to/from remote storage,
primarily AWS S3. It includes functionality for downloading, uploading, and managing cached
processed datasets.
"""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex

from pydantic import BaseModel

import czbenchmarks.cli.utils as cli_utils
from czbenchmarks import exceptions, utils
from czbenchmarks.constants import PROCESSED_DATASETS_CACHE_PATH
from czbenchmarks.datasets import utils as dataset_utils
from czbenchmarks.datasets.dataset import Dataset


log = logging.getLogger(__name__)


class CacheOptions(BaseModel):
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


def get_result_url_for_remote(remote_prefix_url: str) -> str:
    """Generate a unique URL for storing results in remote cache."""
    nonce = token_hex(4)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version = cli_utils.get_version()
    return f"{remote_prefix_url.rstrip('/')}/{version}/results/{timestamp}-{nonce}.json"


def set_processed_datasets_cache(
    dataset: Dataset,
    dataset_name: str,
    *,
    model_name: str,  # TODO: consider to fetch model embedding
    # model_args: ModelArgsDict,
    cache_options: CacheOptions,
) -> None:
    """
    Write a dataset to the cache
    A "processed" dataset has been run with model inference for the given arguments.
    """
    dataset_filename = get_processed_dataset_cache_filename(
        dataset_name,
        model_name=model_name,
        # model_args=model_args
    )
    cache_dir = Path(PROCESSED_DATASETS_CACHE_PATH).expanduser().absolute()
    cache_file = cache_dir / dataset_filename

    try:
        # "Unload" the source data so we only cache the results
        dataset.unload_data()
        cache_dir.mkdir(parents=True, exist_ok=True)
        dataset.serialize(str(cache_file))
        succeeded = True
    except Exception as e:
        # Log the exception, but don't raise if we can't write to the cache for some reason
        log.exception(
            f'Failed to serialize processed dataset to cache "{cache_file}": {e}'
        )
        succeeded = False

    if succeeded and cache_options.upload_embeddings:
        # upload the new embeddings, overwriting any that may already exist
        remote_prefix = get_remote_cache_prefix(cache_options)
        try:
            utils.upload_file_to_remote(
                cache_file, remote_prefix, overwrite_existing=True
            )
            log.info(f"Uploaded processed dataset from {cache_file} to {remote_prefix}")
        except exceptions.RemoteStorageError:
            log.exception("Unable to upload processed dataset to remote cache")

    dataset.load_data()


def try_processed_datasets_cache(
    dataset_name: str,
    *,
    model_name: str,  # TODO: consider to fetch model embedding
    # model_args: ModelArgsDict,
    cache_options: CacheOptions,
) -> Dataset | None:
    """
    Deserialize and return a processed dataset from the cache if it exists, else return None.
    """
    dataset_filename = get_processed_dataset_cache_filename(
        dataset_name,
        model_name=model_name,
        # model_args=model_args
    )
    cache_dir = Path(PROCESSED_DATASETS_CACHE_PATH).expanduser().absolute()
    cache_file = cache_dir / dataset_filename

    if cache_options.download_embeddings:
        # check the remote cache and download the file if a local version doesn't
        # exist, or if the remote version is newer than the local version
        remote_url = f"{get_remote_cache_prefix(cache_options)}{dataset_filename}"

        local_modified: datetime | None = None
        remote_modified: datetime | None = None
        if cache_file.exists():
            local_modified = datetime.fromtimestamp(
                cache_file.stat().st_mtime, tz=timezone.utc
            )
        try:
            remote_modified = utils.get_remote_last_modified(
                remote_url, make_unsigned_request=False
            )
        except exceptions.RemoteStorageError:
            # not a great way to handle this, but maybe the cache bucket is not public
            try:
                log.warning(
                    "Unsigned request to remote storage cache failed. Trying signed request."
                )
                remote_modified = utils.get_remote_last_modified(
                    remote_url, make_unsigned_request=True
                )
            except exceptions.RemoteStorageError:
                pass
        if remote_modified is None:
            log.info("Remote cached embeddings don't exist. Skipping download.")
        elif local_modified is not None and (remote_modified <= local_modified):
            log.info(
                f"Remote cached embeddings modified at {remote_modified}. "
                f"Local cache files modified more recently at {local_modified}. "
                "Skipping download."
            )
        else:
            try:
                utils.download_file_from_remote(remote_url, cache_dir)
                log.info(
                    f"Downloaded cached embeddings from {remote_url} to {cache_dir}"
                )
            except exceptions.RemoteStorageError:
                # not a great way to handle this, but maybe the cache bucket is not public
                try:
                    log.warning(
                        "Unsigned request to remote storage cache failed. Trying signed request."
                    )
                    utils.download_file_from_remote(
                        remote_url, cache_dir, make_unsigned_request=False
                    )
                    log.info(
                        f"Downloaded cached embeddings from {remote_url} to {cache_dir}"
                    )
                except exceptions.RemoteStorageError:
                    log.warning(
                        f"Unable to retrieve embeddings from remote cache at {remote_url!r}"
                    )

    if cache_file.exists():
        # Load the original dataset
        dataset = dataset_utils.load_dataset(dataset_name)
        dataset.load_data()

        # Attach the cached results to the dataset
        processed_dataset = Dataset.deserialize(str(cache_file))
        dataset._outputs = processed_dataset._outputs
        return dataset

    return None


def get_remote_cache_prefix(cache_options: CacheOptions) -> str:
    """Get the prefix ending in '/' that the remote processed datasets go under."""
    return f"{cache_options.remote_cache_url.rstrip('/')}/{cli_utils.get_version()}/processed-datasets/"


def get_processed_dataset_cache_filename(
    dataset_name: str,
    *,
    model_name: str,
    # model_args: ModelArgsDict
) -> str:
    """
    Generate a unique filename for the given dataset and model arguments.
    """
    # if model_args:
    #     model_args_str = f"{model_name}_" + "_".join(
    #         f"{k}-{v}" for k, v in sorted(model_args.items())
    #     )
    # else:
    model_args_str = model_name
    filename = f"{dataset_name}_{model_args_str}.dill"
    return filename


def get_processed_dataset_cache_path(
    dataset_name: str,
    *,
    model_name: str,
    #  model_args: ModelArgsDict
) -> Path:
    """
    Return a unique file path in the cache directory for the given dataset and model arguments.
    """
    cache_dir = Path(PROCESSED_DATASETS_CACHE_PATH).expanduser().absolute()
    filename = get_processed_dataset_cache_filename(
        dataset_name,
        model_name=model_name,
        # model_args=model_args
    )
    return cache_dir / filename