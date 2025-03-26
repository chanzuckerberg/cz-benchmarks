import os
from datetime import datetime

import boto3
from botocore.config import Config
import botocore
import logging

logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("botocore.httpchecksum").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def get_aws_credentials(profile="default", aws_shared_credentials_path=None):
    """
    Get AWS credentials from the specified profile.

    :param profile: AWS profile name (default: 'default')
    :param aws_shared_credentials_path: Path to the AWS credentials file (optional)
    :return: Dictionary containing AWS credentials
    """

    # Check environment variable and default path for credentials file
    if aws_shared_credentials_path is None:
        default_aws_shared_credentials_path = "~/.aws/credentials"
        aws_shared_credentials_path = os.environ.get(
            "AWS_SHARED_CREDENTIALS_FILE", default_aws_shared_credentials_path
        )

    aws_shared_credentials_path = os.path.expanduser(aws_shared_credentials_path)
    if not os.path.exists(aws_shared_credentials_path):
        logger.warning(
            f"AWS credentials file not found at {aws_shared_credentials_path}. "
            f"This may cause issues when accessing AWS resources."
        )
        credentials_dict = {}
    else:
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = aws_shared_credentials_path
        session = boto3.Session(profile_name=profile)
        credentials = session.get_credentials()

        # See if these are valid credentials
        if hasattr(credentials, "access_key") and hasattr(credentials, "secret_key"):
            logger.info(
                f"Using AWS credentials found at {aws_shared_credentials_path}."
            )
            credentials_dict = {
                "AWS_ACCESS_KEY_ID": credentials.access_key,
                "AWS_SECRET_ACCESS_KEY": credentials.secret_key,
            }
        else:
            logger.warning(
                f"Entries for both aws_access_key_id and aws_secret_access_key"
                f" were not found for profile {profile} at"
                f" {aws_shared_credentials_path}. AWS credentials will"
                f" not be added to the container."
            )
            credentials_dict = {}

    return credentials_dict


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
