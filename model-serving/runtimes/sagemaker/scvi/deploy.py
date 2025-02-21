# This file is used to create a model and endpoint in SageMaker
# It also creates an IAM role if it doesn't exist, which is used by SageMaker to access S3 and other AWS services

from datetime import datetime
import logging
import sagemaker
from sagemaker.pytorch.model import PyTorchModel
from utils import create_sagemaker_execution_role, model_exists, endpoint_exists, endpoint_config_exists, create_async_endpoint_config
import boto3

# Set constants
REGION = "us-west-2"
BUCKET = "omar-data"
MODEL_NAME = "scvi"
ENDPOINT_NAME = "scvi-endpoint"

logger = logging.getLogger(__name__)

def main():
    # Create IAM role if it doesn't exist
    role = create_sagemaker_execution_role("OmarSageMakerRole")

    # Create SageMaker client and session
    sm_client = boto3.client("sagemaker", region_name=REGION)
    sm_session = sagemaker.Session()
    
    # Determine if model and endpoint already exist
    model_already_exists = model_exists(MODEL_NAME, sm_client)
    endpoint_already_exists = endpoint_exists(ENDPOINT_NAME, sm_client)

    # Replace model if it exists
    if model_already_exists:
        print(f"Model '{MODEL_NAME}' already exists. Deleting the existing model.")
        # TODO: Upload a model package and register a new version of the model to AWS SageMaker Model Registry 
        # So we can update the model without deleting and recreating the entire model
        sm_client.delete_model(ModelName=MODEL_NAME)
    else:
        print(f"Model '{MODEL_NAME}' does not exist. Creating a new model.")

    # Create model
    pytorch_model = PyTorchModel(
        model_data=f"s3://{BUCKET}/scvi/scvi_model_code.tar.gz",
        role=role,
        framework_version="2.5",
        py_version="py311",
        entry_point="inference.py",
        source_dir="code/",
        sagemaker_session=sm_session,
        name=f"{MODEL_NAME}"
    )

    pytorch_model.create(instance_type="ml.g4dn.xlarge")
    print(f"Model '{MODEL_NAME}' has been created or updated.")

    endpoint_config_name = f"{ENDPOINT_NAME}-config-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    create_async_endpoint_config(endpoint_config_name, MODEL_NAME)

    if endpoint_already_exists:
        print(f"Endpoint '{ENDPOINT_NAME}' already exists. Updating the existing endpoint.")
        response = sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name
        )
        logger.info(f"Endpoint '{ENDPOINT_NAME}' updated with arn: {response['EndpointArn']}")
    else:
        print(f"Endpoint '{ENDPOINT_NAME}' does not exist. Creating a new endpoint.")
        response = sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name
        )
        logger.info(f"Endpoint '{ENDPOINT_NAME}' created with arn: {response['EndpointArn']}")

    # Create a waiter for the endpoint to be in service
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=ENDPOINT_NAME)
    logger.info(f"Endpoint '{ENDPOINT_NAME}' is in service")

    # Log the URL of the endpoint
    endpoint_url = f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{ENDPOINT_NAME}/invocations"
    print(f"Endpoint URL: {endpoint_url}")

if __name__ == "__main__":
    main()
