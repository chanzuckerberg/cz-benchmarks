# This file is used to create a model and endpoint in SageMaker
# It also creates an IAM role if it doesn't exist, which is used by SageMaker to access S3 and other AWS services

from datetime import datetime
import logging
import uuid
import sagemaker
from sagemaker.pytorch.model import PyTorchModel
from utils import (
    create_sagemaker_execution_role,
    endpoint_exists,
    create_async_endpoint_config,
)
import boto3
from botocore.exceptions import ClientError

REGION = "us-west-2"
BUCKET = "omar-data"
MODEL_NAME = "scvi"
ENDPOINT_NAME = "scvi-endpoint"

logger = logging.getLogger(__name__)

def main():
    role = create_sagemaker_execution_role("OmarSageMakerRole")
    sm_client, sm_session = setup_sagemaker_session(REGION)
    
    # Create PyTorch Model
    create_pytorch_model(sm_session, role)
    
    # Create Model Package Group
    model_package_group_name = "scvi-model-package-group"
    create_model_package_group(sm_client, model_package_group_name, "Model Package Group for SCVI")

    # Register Model to Package Group
    register_model_package(sm_client, model_package_group_name)
    
    # Endpoint Configuration and Deployment
    endpoint_config_name = create_endpoint_configuration(ENDPOINT_NAME, MODEL_NAME)
    create_or_update_endpoint(sm_client, ENDPOINT_NAME, endpoint_config_name)
    
    # Wait for the endpoint to be in service 
    wait_for_endpoint_in_service(sm_client, ENDPOINT_NAME)

    # Log the endpoint URL
    endpoint_url = f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{ENDPOINT_NAME}/invocations"
    logger.info(f"Endpoint URL: {endpoint_url}")

def setup_sagemaker_session(region: str):
    """Create SageMaker client and session."""
    sm_client = boto3.client("sagemaker", region_name=region)
    sm_session = sagemaker.Session()
    return sm_client, sm_session

def create_pytorch_model(sm_session, role: str) -> PyTorchModel:
    """Create or update the PyTorch model in SageMaker."""
    # Under the hood, the model is using AWS managed image: 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.5-gpu-py311
    pytorch_model = PyTorchModel(
        model_data=f"s3://{BUCKET}/scvi/scvi_model_code.tar.gz",
        role=role,
        framework_version="2.5",
        py_version="py311",
        entry_point="inference.py",
        source_dir="code/",
        sagemaker_session=sm_session,
        name=MODEL_NAME
    )
    pytorch_model.create(instance_type="ml.g4dn.xlarge")
    logger.info(f"Model '{MODEL_NAME}' has been created or updated.")
    return pytorch_model

def create_model_package_group(sm_client, group_name, description):
    """
    Creates a SageMaker model package group if it does not already exist.

    Parameters:
        sm_client (boto3.client): The SageMaker client.
        group_name (str): The name of the model package group.
        description (str): A description for the model package group.

    Returns:
        str: The ARN of the created or existing model package group.
    """
    try:
        response = sm_client.describe_model_package_group(ModelPackageGroupName="scvi-model-package-group")
        model_package_group_arn = response['ModelPackageGroupArn']
        logger.info(f"Retrieved existing Model Package Group ARN: {model_package_group_arn}")
        return model_package_group_arn
    except ClientError:
        response = sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=description
        )
        model_package_group_arn = response['ModelPackageGroupArn']
        logger.info(f"Model Package Group '{group_name}' created successfully with ARN: {model_package_group_arn}")
        return model_package_group_arn

def register_model_package(sm_client, model_package_group_name: str, model_package_name: str = None):
    """Register the model to a SageMaker Model Package Group."""
    if not model_package_name:
        model_package_name = f"scvi-package-{uuid.uuid4()}"
    
    try:
        response = sm_client.create_model_package(
            ModelPackageGroupName=model_package_group_name,
            ModelPackageDescription="SCVI model package",
            InferenceSpecification={
                'Containers': [{
                    'Image': '763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.5-gpu-py311',
                    'ModelDataUrl': f"s3://{BUCKET}/scvi/scvi_model_code.tar.gz"
                }],
                'SupportedTransformInstanceTypes': ['ml.g4dn.xlarge'],
                'SupportedContentTypes': ['application/json'],
                'SupportedResponseMIMETypes': ['application/json']
            },
            CertifyForMarketplace=False
        )
        logger.info(f"Model Package '{model_package_name}' created and added to Group '{model_package_group_name}'.")
        return response['ModelPackageArn']
    except Exception as e:
        logger.error(f"Failed to create Model Package: {e}")
        raise

def create_endpoint_configuration(endpoint_name: str, model_name: str) -> str:
    """Create an asynchronous endpoint configuration."""
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    endpoint_config_name = f"{endpoint_name}-config-{timestamp}"
    create_async_endpoint_config(endpoint_config_name, model_name)
    return endpoint_config_name

def create_or_update_endpoint(sm_client, endpoint_name: str, endpoint_config_name: str):
    """Create a new endpoint or update an existing one."""
    if endpoint_exists(endpoint_name, sm_client):
        logger.info(f"Endpoint '{endpoint_name}' already exists. Updating the existing endpoint.")
        response = sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        logger.info(f"Endpoint '{endpoint_name}' updated with arn: {response['EndpointArn']}")
    else:
        logger.info(f"Endpoint '{endpoint_name}' does not exist. Creating a new endpoint.")
        response = sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        logger.info(f"Endpoint '{endpoint_name}' created with arn: {response['EndpointArn']}")

def wait_for_endpoint_in_service(sm_client, endpoint_name: str):
    """Wait for the endpoint to be in service."""
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    logger.info(f"Endpoint '{endpoint_name}' is in service")

if __name__ == "__main__":
    main()
