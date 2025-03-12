import os
from datetime import datetime

import boto3
import logging

logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("botocore.httpchecksum").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def get_aws_credentials(profile="default"):
    """
    Get AWS credentials from the specified profile.

    :param profile: AWS profile name (default: 'default')
    :return: Dictionary containing AWS credentials
    """
    # Create a session with the specified profile
    session = boto3.Session(profile_name=profile)

    # Get credentials from the session
    credentials = session.get_credentials()

    if not (hasattr(credentials, "access_key") and hasattr(credentials, "secret_key")):
        raise ValueError(
            f"AWS credentials not found for profile {profile}."
            " Entries for aws_access_key_id and aws_secret_access_key"
            " must be present in the profile."
        )

    return {
        "AWS_ACCESS_KEY_ID": credentials.access_key,
        "AWS_SECRET_ACCESS_KEY": credentials.secret_key,
    }


def download_s3_file(bucket, key, local_path):
    """
    Downloads a single file from S3 to a local path.

    :param bucket: S3 bucket name
    :param key: S3 key (file path) to download
    :param local_path: Local file path to save to
    """
    s3 = boto3.client("s3")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        s3.download_file(bucket, key, local_path)
        logger.info(f"Downloaded {key} from s3://{bucket} to {local_path}")
    except Exception as e:
        logger.error(f"Failed to download {key} from s3://{bucket}: {str(e)}")
        raise


def sync_s3_to_local(bucket, prefix, local_dir):
    """
    Syncs files from an S3 bucket prefix to a local directory.

    :param bucket: S3 bucket name
    :param prefix: S3 prefix (directory path or file) to sync from
    :param local_dir: Local directory path to sync to
    """
    s3 = boto3.client("s3")

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
