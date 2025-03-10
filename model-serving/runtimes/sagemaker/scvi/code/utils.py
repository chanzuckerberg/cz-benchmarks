"""
Utilities for the SCVI service.
"""

import os
import logging
import boto3
import base64
import numpy as np

logger = logging.getLogger(__name__)


def download_from_s3(s3_path: str, local_path: str = None) -> str:
    """
    Download a file from S3 and return the local file path.

    :param s3_path: S3 URI (e.g. "s3://bucket/path/to/file.h5ad")
    :param local_path: Optional local path to save the file. Defaults to /tmp/<filename>.
    :return: Local file path.
    """
    logger.info("S3 path: %s", s3_path)
    if local_path is None:
        local_path = os.path.join("/tmp", os.path.basename(s3_path))
    try:
        s3_client = boto3.client("s3")
        # Remove the "s3://" prefix and split into bucket and key.
        s3_uri = s3_path.replace("s3://", "")
        bucket, key = s3_uri.split("/", 1)
        logger.info("Downloading file from S3: bucket=%s, key=%s", bucket, key)
        s3_client.download_file(bucket, key, local_path)
        return local_path
    except Exception as e:
        logger.error("Error downloading file from S3: %s", e)
        raise


def encode_array(array):
    # If it's a torch tensor, convert to numpy
    if hasattr(array, "detach") and callable(array.detach):
        array = array.detach().cpu().numpy()

    # Ensure it's numpy & convert to bytes
    arr_np = np.array(array)
    arr_bytes = arr_np.tobytes()

    # Encode in base64
    arr_b64 = base64.b64encode(arr_bytes).decode("utf-8")

    # Return shape, dtype, and the encoded data for easy reconstruction
    return {"shape": arr_np.shape, "dtype": str(arr_np.dtype), "data_b64": arr_b64}


def decode_array(encoded):
    arr_shape = tuple(encoded["shape"])
    arr_dtype = encoded["dtype"]
    arr_b64 = encoded["data_b64"]
    arr_bytes = base64.b64decode(arr_b64)

    # Reconstruct numpy array
    array = np.frombuffer(arr_bytes, dtype=arr_dtype).reshape(arr_shape)
    return array
