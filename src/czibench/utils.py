import os
from datetime import datetime

import boto3


def sync_s3_to_local(bucket, prefix, local_dir):
    """
    Syncs files from an S3 bucket prefix to a local directory.

    :param bucket: S3 bucket name
    :param prefix: S3 prefix (directory path) to sync from
    :param local_dir: Local directory path to sync to
    """
    s3 = boto3.client("s3")
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
                print(f"Downloaded: {relative_key}")
            else:
                # Compare last modified times
                s3_time = obj["LastModified"].replace(tzinfo=None)
                local_time = datetime.utcfromtimestamp(
                    os.path.getmtime(local_file_path)
                )
                if s3_time > local_time:
                    s3.download_file(bucket, key, local_file_path)
                    print(f"Updated: {relative_key}")
