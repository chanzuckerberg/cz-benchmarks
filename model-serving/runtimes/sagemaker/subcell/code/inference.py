# This file is used to define the necessary functions to deploy the Subcell model to SageMaker
# model_fn, input_fn, predict_fn, output_fn are required to be implemented by SageMaker
# More info on the dir structure: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#model-directory-structure
# These get packaged up into a .tar.gz file, uploaded to an S3 bucket, and then deployed to SageMaker via deploy.py

# See also https://github.com/CellProfiling/SubCellPortable/blob/main/process.py for where much of this logic was adapted from.

import json
from omegaconf import OmegaConf
from utils import download_from_s3
import os
import logging
from skimage.io import imread
import numpy as np
from io import BytesIO

from vit_model import ViTPoolClassifier
import inference_utils

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    """
    Load and initialize the Subcell model from the specified directory.

    This function is invoked by SageMaker to deserialize and load the Subcell model
    during deployment. It expects a `config.yaml` file within the `model_dir` that
    outlines the necessary artifact paths required for the model's operation.

    Args:
        model_dir (str): The directory path where the model artifacts are stored.

    Returns:
        Subcell: An instance of the Subcell model initialized with the loaded artifacts.

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

    # TODO: Allow multiple classifiers
    classifier_path = artifacts.get("classifier_paths")[0]
    logger.info("Downloading classifier from S3")
    classifier_local_path = download_from_s3(classifier_path)
    encoder_path = artifacts.get("encoder_path")
    logger.info("Downloading encoder from S3")
    encoder_local_path = download_from_s3(encoder_path)

    model_config = artifacts.get("model_config")
    model = ViTPoolClassifier(model_config)
    model.load_model_dict(encoder_local_path, [classifier_local_path])
    
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and preprocess the incoming request for prediction.

    This function is called by SageMaker to parse and process each incoming inference request.
    It handles both synchronous and asynchronous requests by accepting a request body and its
    content type. The function expects the request body to be in JSON format with specific
    keys indicating the S3 location of the input data.

    Args:
        request_body (bytes): The raw request payload sent by the client.
        request_content_type (str): The MIME type of the incoming request.

    Returns:
        dict: A dictionary containing the preprocessed image data as numpy arrays, along with the desired output path.

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
    r_image_path = data.get("r_image")
    g_image_path = data.get("g_image")
    y_image_path = data.get("y_image")
    b_image_path = data.get("b_image")
    output_folder = data.get("output_folder")
    output_path = data.get("output_path")

    # TODO: Support local file usage also

    cell_data = []
    if r_image_path:
        image = download_from_s3(r_image_path)
        cell_data.append([imread(image, as_gray=True)])
    if g_image_path:
        image = download_from_s3(g_image_path)
        cell_data.append([imread(image, as_gray=True)])
    if y_image_path:
        image = download_from_s3(y_image_path)
        cell_data.append([imread(image, as_gray=True)])
    if b_image_path:
        image = download_from_s3(b_image_path)
        cell_data.append([imread(image, as_gray=True)])

    cell = {"cell_data": cell_data, "output_folder": output_folder, "output_path": output_path}

    return cell


def predict_fn(input_data, model):
    """
    Execute the prediction logic using the loaded Subcell model.

    This function takes the preprocessed input data and the loaded Subcell model to perform
    inference. It handles delegating the prediction task to the model and, if a classifier model(s) was provided,
    getting the top 3 predicted location names.

    Args:
        input_data (dict): A dictionary containing the image data as numpy arrays and output path.
        model: The loaded Subcell model instance.

    Returns:
        dict: A dictionary containing the embeddings, probabilities, and top 3 location names predicted by the model.

    Raises:
        KeyError: If required keys are missing in the `input_data` dictionary.
        FileNotFoundError: If the necessary model weights are not found.
    """
    try:
        cell_data = input_data["cell_data"]
        output_folder = input_data["output_folder"]
        output_path = input_data["output_path"]
    except KeyError as e:
        raise KeyError(f"Missing key in input_data: {e}")

    # Delegate to model's predict method
    logger.info(f"Predicting")
    # This is where the embedding, probabilities, and attention map files will be stored.
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_path)

    embedding, probabilities = inference_utils.run_model(model, cell_data, output_path)
    # If at least one classifier model was provided, get the top 3 predicted locations
    curr_probs_l = probabilities.tolist()
    max_location_class = curr_probs_l.index(max(curr_probs_l))
    max_location_name = inference_utils.CLASS2NAME[max_location_class]
    max_3_location_classes = sorted(
        range(len(curr_probs_l)), key=lambda sub: curr_probs_l[sub]
    )[-3:]
    max_3_location_classes.reverse()
    max_3_location_names = (
        inference_utils.CLASS2NAME[max_3_location_classes[0]]
        + ","
        + inference_utils.CLASS2NAME[max_3_location_classes[1]]
        + ","
        + inference_utils.CLASS2NAME[max_3_location_classes[2]]
    )

    result = {"embedding": embedding, "probabilities": probabilities, "max_3_location_names": max_3_location_names}
    return result


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
    logger.info(f"Prediction: {prediction}")

    if content_type == "application/x-npy":
        # Convert the prediction from a dictionary to a NumPy array.
        prediction_array = np.fromiter(prediction.items(), dtype="object", count=len(prediction))

        # Convert NumPy array to binary stream in NumPy .npy format
        buffer = BytesIO()
        np.save(buffer, prediction_array)
        return buffer.getvalue(), "application/x-npy"
        # When making a request to the endpoint, the response is a binary stream in NumPy .npy format so we can reconstruct the array using np.load(io.BytesIO(response_body)

    elif content_type == "application/json":
        # Convert dict to JSON
        return json.dumps(prediction), "application/json"

    else:
        # Default to NumPy binary format if content_type is unsupported
        prediction_array = np.fromiter(prediction.items(), dtype="object", count=len(prediction))
        buffer = BytesIO()
        np.save(buffer, prediction_array)
        return buffer.getvalue(), "application/x-npy"
