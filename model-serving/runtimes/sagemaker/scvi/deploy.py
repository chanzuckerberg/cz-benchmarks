# This file is used to create a model and endpoint in SageMaker
# It also creates an IAM role if it doesn't exist, which is used by SageMaker to access S3 and other AWS services

from datetime import datetime
import sagemaker
from sagemaker.async_inference import AsyncInferenceConfig
from sagemaker.pytorch.model import PyTorchModel
from utils import create_sagemaker_execution_role, model_exists, endpoint_exists, endpoint_config_exists
import boto3

# Set constants
REGION = "us-west-2"
BUCKET = "omar-data"
MODEL_NAME = "scvi"
ENDPOINT_NAME = "scvi-endpoint"
MODEL_DATA = f"s3://{BUCKET}/scvi/scvi_model_code.tar.gz"

def main():
    # Create IAM role if it doesn't exist
    role = create_sagemaker_execution_role("OmarSageMakerRole")

    # Create SageMaker client and session
    sm_client = boto3.client("sagemaker", region_name=REGION)
    sm_session = sagemaker.Session()
    
    # Determine if model and endpoint already exist
    model_already_exists = model_exists(MODEL_NAME, sm_client)
    endpoint_already_exists = endpoint_exists(ENDPOINT_NAME, sm_client)
    endpoint_config_already_exists = endpoint_config_exists(ENDPOINT_NAME, sm_client)

    # Replace model if it exists
    if model_already_exists:
        print(f"Model '{MODEL_NAME}' already exists. Deleting the existing model.")
        sm_client.delete_model(ModelName=MODEL_NAME)
    else:
        print(f"Model '{MODEL_NAME}' does not exist. Creating a new model.")

    # Create model
    pytorch_model = PyTorchModel(
        model_data=MODEL_DATA,
        role=role,
        framework_version="2.5",
        py_version="py311",
        entry_point="inference.py",
        source_dir="code/",
        sagemaker_session=sm_session,
        name=f"{MODEL_NAME}"
    )

    print(f"Model '{MODEL_NAME}' has been created or updated.")

    # Create endpoint config and deploy model
    if endpoint_config_already_exists:
        print(f"Endpoint config '{ENDPOINT_NAME}' already exists. Deleting the existing endpoint config.")
        sm_client.delete_endpoint_config(EndpointConfigName=ENDPOINT_NAME)
    else:
        print(f"Endpoint config '{ENDPOINT_NAME}' does not exist. Creating a new endpoint config.")
    
    async_config = AsyncInferenceConfig(
        output_path=f"s3://{BUCKET}/scvi-async-output/",
        failure_path=f"s3://{BUCKET}/scvi-async-failure/",
        # TODO: Add notification configs for success and failure
    )

    # Replace endpoint if it exists
    if endpoint_already_exists:
        print(f"Endpoint '{ENDPOINT_NAME}' already exists. Deleting the existing endpoint.")
        sm_client.delete_endpoint(EndpointName=ENDPOINT_NAME)
    else:
        print(f"Endpoint '{ENDPOINT_NAME}' does not exist. Creating a new endpoint.")

    pytorch_model.deploy(
        initial_instance_count=1,
        instance_type="ml.g4dn.xlarge",
        async_inference_config=async_config,
        endpoint_name=ENDPOINT_NAME
    )

    # Log the URL of the endpoint
    endpoint_url = f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{ENDPOINT_NAME}/invocations"
    print(f"Endpoint URL: {endpoint_url}")

if __name__ == "__main__":
    main()
