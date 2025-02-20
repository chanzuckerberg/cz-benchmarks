import functools
import json
from anndata import AnnData, read_h5ad
from omegaconf import OmegaConf
import scvi
import anndata as ad
import pandas as pd
from scipy import sparse
from utils import download_from_s3
import os
import logging
from pathlib import Path
import os
import yaml
import numpy as np
from utils import encode_array
from io import BytesIO

logger = logging.getLogger(__name__)


# model_fn, input_fn, predict_fn, output_fn are required by SageMaker
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


def predict_fn(input_data, model):
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
    except KeyError as e:
        raise KeyError(f"Missing key in input_data: {e}")

    logger.info(f"Downloading model weights")
    # Download model weights from S3 for the specified organism
    model._download_model_weights(organism)
    model_dir = model.artifacts.get(f"model_weights_{organism}")

    # Filter adata by HVGs for the specified organism
    logger.info(f"Filtering adata by HVGs")
    hvg_file = model.artifacts.get(f"hvg_names_{organism}")
    adata = model._filter_adata_by_hvg(adata, organism)

    # Delegate to model's predict method
    logger.info(f"Predicting")
    return model._predict(adata, hvg_file, model_dir)


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


class SCVI:
    """
    This class loads scvi-tools models and performs inference.
    It expects a dictionary of artifact paths in self.artifacts.
    """

    def __init__(self, artifacts: dict = None):
        # Initialize the SCVI model. If artifacts are not provided, load them from config.yaml
        if artifacts is None:
            config_path = os.environ.get("SCVI_CONFIG", "config.yaml")
            with open(config_path, "r") as f:
                artifacts = yaml.safe_load(f)
        self.artifacts = artifacts

    @staticmethod
    def predict(adata: AnnData, hvg_file: str, reference_model_path: Path):
        batch_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]

        # Filter input anndata by HVGs
        logger.info(f"Filtering adata by HVGs")
        adata = SCVI._filter_adata_by_hvg(adata, hvg_file)

        adata.obs["batch"] = functools.reduce(
            lambda a, b: a + b, [adata.obs[c].astype(str) for c in batch_keys]
        )

        logger.info(f"Preparing query anndata")
        scvi.model.SCVI.prepare_query_anndata(
            adata, str(reference_model_path), return_reference_var_names=True
        )
        logger.info(f"Loading query data")
        vae_q = scvi.model.SCVI.load_query_data(
            adata,
            str(reference_model_path),
        )
        vae_q.is_trained = True

        # Get latent representation
        logger.info(f"Getting latent representation")
        qz_m, _ = vae_q.get_latent_representation(return_dist=True)
        logger.info(f"Latent representation shape: {qz_m.shape}")
        return qz_m

    def _filter_adata_by_hvg(self, adata: ad.AnnData, organism: str) -> ad.AnnData:
        """Filter adata by HVGs for the specified organism, downloading HVG names if not already present."""

        # Check if HVG file for the organism is already downloaded
        hvg_key = f"hvg_names_{organism}"
        if hvg_key not in self.artifacts:
            logger.info(f"HVG file for organism '{organism}' not found. Downloading...")
            self._download_hvg_names(organism)

        hvg_path = self.artifacts.get(hvg_key)
        if not hvg_path or not os.path.exists(hvg_path):
            raise FileNotFoundError(
                f"HVG file for organism '{organism}' could not be found or downloaded."
            )

        adata = adata.copy()

        hvg = pd.read_csv(hvg_path)
        adata.var["feature_id"] = adata.var["feature_id"].astype(str)
        hvg["feature_id"] = hvg["feature_id"].astype(str)

        # Remove duplicate columns if present
        adata.var = adata.var.loc[:, ~adata.var.columns.duplicated()]

        mask = adata.var["feature_id"].isin(hvg["feature_id"])
        adata_filtered = adata[:, mask].copy()

        missing_features = set(hvg.feature_id) - set(adata.var.feature_id)
        if missing_features:
            logger.info(
                f"WARNING: {len(missing_features)} HVGs are not present in the AnnData object"
            )
            missing_var = pd.DataFrame({"feature_id": list(missing_features)})
            missing_var["feature_name"] = missing_var["feature_id"]
            missing_var.set_index("feature_name", inplace=True)
            missing_X = sparse.csr_matrix((adata.n_obs, len(missing_features)))
            adata_missing = ad.AnnData(
                X=missing_X, var=missing_var, obs=adata_filtered.obs.copy()
            )
            adata_concat = ad.concat(
                [adata_filtered, adata_missing], axis=1, join="outer", merge="first"
            )
        else:
            adata_concat = adata_filtered

        hvg_unique = hvg.drop_duplicates(subset="feature_id")
        adata_reordered = ad.AnnData(
            X=adata_concat[:, hvg_unique.feature_id].X,
            obs=adata_concat.obs.copy(),
            var=adata_concat.var.loc[hvg_unique.feature_id].copy(),
        )
        return adata_reordered

    def _download_hvg_names(self, organism: str):
        """
        Downloads HVG names from S3 for the specified organism.

        Args:
            organism (str): The organism identifier (e.g., 'homo_sapiens').
        """
        hvg_val = self.artifacts.get(organism, {}).get("hvg_names")
        if hvg_val and isinstance(hvg_val, str) and hvg_val.startswith("s3://"):
            local_dir = "/tmp"
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, os.path.basename(hvg_val))
            if not os.path.exists(local_path):
                download_from_s3(hvg_val, local_path)
            self.artifacts[f"hvg_names_{organism}"] = local_path
            self.artifacts[organism]["hvg_names"] = local_path

    def _download_model_weights(self, organism: str):
        """
        Downloads model weights from S3 for the specified organism.

        Args:
            organism (str): The organism identifier (e.g., 'homo_sapiens').
        """
        mw_val = self.artifacts.get(organism, {}).get("model_weights")
        if mw_val and isinstance(mw_val, str) and mw_val.startswith("s3://"):
            local_dir = os.environ.get("MODEL_DIR", "./model_weights")
            os.makedirs(local_dir, exist_ok=True)
            local_path = os.path.join(local_dir, os.path.basename(mw_val))
            if not os.path.exists(local_path):
                download_from_s3(mw_val, local_path)
            self.artifacts[f"model_weights_{organism}"] = local_path
            self.artifacts[organism]["model_weights"] = local_path
