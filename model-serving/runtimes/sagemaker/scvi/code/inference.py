# This file is used to define the necessary functions to deploy the SCVI model to SageMaker
# model_fn, input_fn, predict_fn, output_fn are required to be implemented by SageMaker
# More info on the dir structure: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#model-directory-structure
# These get packaged up into a .tar.gz file, uploaded to an S3 bucket, and then deployed to SageMaker via deploy.py

import json
from anndata import read_h5ad
from omegaconf import OmegaConf
from utils import download_from_s3
import os
import logging
import os
import numpy as np
from io import BytesIO
from scvi_model import SCVI

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """
    Load and initialize the SCVI model from the specified directory.

    This function is invoked by SageMaker to deserialize and load the SCVI model
    during deployment. It expects a `config.yaml` file within the `model_dir` that
    outlines the necessary artifact paths required for the model's operation.

    Args:
        model_dir (str): The directory path where the model artifacts are stored.

    Returns:
        SCVI: An instance of the SCVI model initialized with the loaded artifacts.

    Raises:
        FileNotFoundError: If the `config.yaml` file is not found in the provided `model_dir`.
    """
    logger.info(f"Model directory: {model_dir}")
    logger.info(f"Contents of model_dir: {os.listdir(model_dir)}")
    config_file = os.path.join(model_dir, "code/config.yaml")

    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found in model_dir: {config_file}")
    with open(config_file, "r") as f:
        cfg = OmegaConf.load(f)
        artifacts = OmegaConf.to_container(cfg, resolve=True)

    model = SCVI(artifacts)
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and preprocess the incoming request for prediction.

    This function is called by SageMaker to parse and process each incoming inference request.
    It handles both synchronous and asynchronous requests by accepting a request body and its
    content type. The function expects the request body to be in JSON format with specific
    keys indicating the S3 location of the input data and the organism type.

    Args:
        request_body (bytes): The raw request payload sent by the client.
        request_content_type (str): The MIME type of the incoming request.

    Returns:
        dict: A dictionary containing the preprocessed `AnnData` object and the `organism` identifier.

    Raises:
        ValueError: If the `request_content_type` is not supported or required keys are missing in the JSON.
        json.JSONDecodeError: If the `request_body` is not valid JSON.
    """
    logger.info(f"Input function called with content type: {request_content_type}")
    logger.info(f"Request body: {request_body}")
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    # Parse the input JSON
    data = json.loads(request_body)
    logger.info(f"Input data: {data}")
    s3_input = data.get("s3_input")
    organism = data.get("organism")

    if not s3_input or not organism:
        raise ValueError("JSON must contain 's3_input' and 'organism' keys.")

    # Download input file from S3
    logger.info(f"Downloading input file from S3")
    local_path = download_from_s3(s3_input)

    # Load the input file as an AnnData object
    logger.info(f"Loading input file as AnnData object")
    adata = read_h5ad(local_path)

    return {"adata": adata, "organism": organism}


def predict_fn(input_data, model: SCVI):
    """
    Execute the prediction logic using the loaded SCVI model.

    This function takes the preprocessed input data and the loaded SCVI model to perform
    inference. It handles downloading necessary model weights, filtering the input data
    based on highly variable genes (HVGs), and delegating the prediction task to the model.

    Args:
        input_data (dict): A dictionary containing the `AnnData` object and `organism` identifier.
                          Expected keys are `adata` (AnnData) and `organism` (str).
        model (SCVI): The loaded SCVI model instance.

    Returns:
        Any: The prediction result obtained from the SCVI model's `_predict` method.
              The exact type depends on the model's implementation.

    Raises:
        KeyError: If required keys are missing in the `input_data` dictionary.
        FileNotFoundError: If the necessary model weights or HVG files are not found.
    """
    try:
        adata = input_data["adata"]
        organism = input_data["organism"]
        destination_dir = f"/tmp/{organism}"
        os.makedirs(destination_dir, exist_ok=True)
    except KeyError as e:
        raise KeyError(f"Missing key in input_data: {e}")

    logger.info(f"Downloading model weights")
    # Download model weights from S3 for the specified organism
    model_dir_path = model._download_model_weights(organism, destination_dir)
    logger.info(f"{organism} model weights downloaded to {model_dir_path}")

    # Download HVG names from S3 for the specified organism
    logger.info(f"Filtering adata by HVGs")
    hvg_file_path = model._download_hvg_names(organism, destination_dir)
    logger.info(f"{organism} HVG file downloaded to {hvg_file_path}")

    # Delegate to model's predict method
    logger.info(f"Predicting")
    return model.predict(adata, hvg_file_path, destination_dir)


def output_fn(prediction, content_type):
    """
    Serialize the prediction output into the desired response format.

    This function converts the prediction result into a format suitable for the client,
    based on the specified `content_type`. It supports both binary NumPy arrays and JSON
    serialization, allowing clients to reconstruct the prediction accurately.

    Args:
        prediction (Any): The prediction result to be serialized. Typically, this is one or more
                          NumPy `ndarray` objects.
        content_type (str): The desired MIME type for the response. Supported types are
                            `'application/x-npy'` for NumPy binary format and `'application/json'`.

    Returns:
        tuple: A tuple containing the serialized prediction and the corresponding content type.

    Raises:
        ValueError: If the specified `content_type` is not supported.
    """
    logger.info(f"Output function called with content type: {content_type}")
    logger.info(f"Prediction type: {type(prediction)}")
    logger.info(f"Prediction shape: {prediction.shape}")

    if content_type == "application/x-npy":
        # Convert NumPy array to binary stream in NumPy .npy format
        buffer = BytesIO()
        np.save(buffer, prediction)
        return buffer.getvalue(), "application/x-npy"
        # When making a request to the endpoint, the response is a binary stream in NumPy .npy format so we can reconstruct the array using np.load(io.BytesIO(response_body)

    elif content_type == "application/json":
        # Convert NumPy array to JSON
        response_body = prediction.tolist()
        return json.dumps(response_body), "application/json"

    else:
        # Default to NumPy binary format if content_type is unsupported
        buffer = BytesIO()
        np.save(buffer, prediction)
        return buffer.getvalue(), "application/x-npy"
