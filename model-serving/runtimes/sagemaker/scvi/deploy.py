# This file is used to create a model and endpoint in SageMaker
# It also creates an IAM role if it doesn't exist, which is used by SageMaker to access S3 and other AWS services

from datetime import datetime
import sagemaker
from sagemaker.pytorch.model import PyTorchModel
from create_sagemaker_role import create_sagemaker_execution_role
import boto3
from botocore.exceptions import ClientError

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

    # Create model if it doesn't exist
    if not model_already_exists:    
        pytorch_model = PyTorchModel(
            model_data=MODEL_DATA,
            role=role,
            framework_version="2.5",
            py_version="py311",
            entry_point="inference.py",
            source_dir="code",
            sagemaker_session=sm_session,
            model_package_name=f"{MODEL_NAME}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        )

        print(f"Model '{MODEL_NAME}' created.")
    else:
        print (f"Model '{MODEL_NAME}' already exists.")

    # Create endpoint if it doesn't exist
    if not endpoint_already_exists:
        async_config = sagemaker.async_inference.AsyncInferenceConfig(
            output_path=f"s3://{BUCKET}/scvi-async-output/"
        )

        # Deploy model
        pytorch_model.deploy(
            initial_instance_count=1,
            instance_type="ml.m5.large",
            async_inference_config=async_config,
            endpoint_name=ENDPOINT_NAME
        )

        # Log the URL of the endpoint
        endpoint_url = f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{ENDPOINT_NAME}/invocations"
        print(f"Endpoint URL: {endpoint_url}")
    else:
        print(f"Endpoint '{ENDPOINT_NAME}' already exists.")


def model_exists(model_name, sm_client):
    try:
        sm_client.describe_model(ModelName=model_name)
        print(f"Model '{model_name}' already exists.")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            # Model does not exist
            print(f"Model '{model_name}' does not exist.")
            return False
        else:
            raise

def endpoint_exists(endpoint_name, sm_client):
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint '{endpoint_name}' already exists.")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            # Endpoint does not exist
            print(f"Endpoint '{endpoint_name}' does not exist.")
            return False
        else:
            raise

if __name__ == "__main__":
    main()
