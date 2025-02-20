import boto3
from botocore.exceptions import ClientError
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

REGION = "us-west-2"

sagemaker_client = boto3.client('sagemaker', region_name=REGION)

def delete_endpoint(endpoint_name):
    """
    Deletes the specified SageMaker endpoint.

    Parameters:
        endpoint_name (str): The name of the endpoint to delete.

    Raises:
        ClientError: If the deletion fails.
    """
    try:
        logger.info(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info(f"Successfully deleted endpoint: {endpoint_name}")
    except ClientError as e:
        logger.error(f"Failed to delete endpoint {endpoint_name}: {e}")
        raise

def delete_model(model_name):
    """
    Deletes the specified SageMaker model.

    Parameters:
        model_name (str): The name of the model to delete.

    Raises:
        ClientError: If the deletion fails.
    """
    try:
        logger.info(f"Deleting model: {model_name}")
        sagemaker_client.delete_model(ModelName=model_name)
        logger.info(f"Successfully deleted model: {model_name}")
    except ClientError as e:
        logger.error(f"Failed to delete model {model_name}: {e}")
        raise

def cleanup_endpoint_and_model(endpoint_name, model_name):
    """
    Deletes both the SageMaker endpoint and model.

    Parameters:
        endpoint_name (str): The name of the endpoint to delete.
        model_name (str): The name of the model to delete.

    Raises:
        ClientError: If either deletion fails.
    """
    delete_endpoint(endpoint_name)
    delete_model(model_name)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python cleanup.py <endpoint_name> <model_name>")
        sys.exit(1)
    endpoint_name = sys.argv[1]
    model_name = sys.argv[2]
    cleanup_endpoint_and_model(endpoint_name, model_name)
