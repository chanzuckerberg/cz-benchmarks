import json
from botocore.exceptions import ClientError
import boto3
import uuid
import logging
from sagemaker.async_inference import AsyncInferenceConfig
import time
from urllib.parse import urlparse

REGION = "us-west-2"
S3_BUCKET = "omar-data"

logger = logging.getLogger(__name__)

def create_async_endpoint_config(endpoint_config_name, model_name):
    sm_client = boto3.client("sagemaker", region_name=REGION)
    if endpoint_config_exists(endpoint_config_name, sm_client):
        print(f"Endpoint config '{endpoint_config_name}' already exists. Creating a new endpoint config.")
    
    async_config = AsyncInferenceConfig(
        output_path=f"s3://{S3_BUCKET}/scvi-async-output/",
        failure_path=f"s3://{S3_BUCKET}/scvi-async-failure/",
        # TODO: Add notification configs for success and failure
    )

    response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "ModelName": model_name,
                "VariantName": "SCVIHumanMouseVariant",
                "InitialInstanceCount": 1,
                "InstanceType": "ml.g4dn.xlarge"
            }
        ],
        AsyncInferenceConfig=async_config._to_request_dict(),
        # ExecutionRoleArn?? 

    )

    endpoint_config_arn = response["EndpointConfigArn"]
    logger.info(f"Endpoint config '{endpoint_config_name}' created with arn: {endpoint_config_arn}")

    return endpoint_config_arn


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


def endpoint_config_exists(endpoint_config_name, sm_client):
    try:
        sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"Endpoint config '{endpoint_config_name}' already exists.")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            # Endpoint config does not exist
            print(f"Endpoint config '{endpoint_config_name}' does not exist.")
            return False
        else:
            raise


def create_sagemaker_execution_role(role_name="SageMakerExecutionRole"):
    """
    Creates an IAM role that SageMaker can assume, with permissions to
    read from S3 and use basic SageMaker features.

    Returns:
        The ARN of the created (or existing) IAM role.
    """
    iam = boto3.client("iam")

    # 1) Define the trust policy so SageMaker can assume the role
    assume_role_policy_document = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {
                    "Service": "sagemaker.amazonaws.com"
                },
                "Action": "sts:AssumeRole"
            }
        ]
    }

    # Check if the role already exists
    try:
        existing_role = iam.get_role(RoleName=role_name)
        print(f"Role '{role_name}' already exists. ARN: {existing_role['Role']['Arn']}")
        return existing_role['Role']['Arn']
    except iam.exceptions.NoSuchEntityException:
        pass  # Role does not exist, proceed to create it

    # 2) Create the role
    print(f"Creating IAM role '{role_name}'...")
    create_role_response = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
        Description="Role for SageMaker to access S3 and other AWS services"
    )
    role_arn = create_role_response["Role"]["Arn"]
    print(f"Created role: {role_arn}")

    # 3) Attach policies. 
    #    *At minimum*, attach AmazonS3FullAccess if you need S3 read and write permissions
    #    and a restricted SageMaker policy (like AmazonSageMakerFullAccess or a custom policy).
    #    For demonstration, we'll attach these AWS-managed policies:
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonS3FullAccess"
    )
    iam.attach_role_policy(
        RoleName=role_name,
        PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
    )

    print(f"Attached AmazonS3FullAccess and AmazonSageMakerFullAccess to {role_name}")

    # Give a little time for the role to fully propagate (sometimes needed)
    import time
    time.sleep(5)

    return role_arn

def upload_to_s3(payload, prefix="scvi-async-input/"):
    s3_client = boto3.client('s3', region_name=REGION)
    inference_id = str(uuid.uuid4())
    s3_key = f"{prefix}{inference_id}.json"
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=payload)
    s3_uri = f"s3://{S3_BUCKET}/{s3_key}"
    return inference_id, s3_uri

def download_s3_file(s3_uri, local_filename):
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    s3 = boto3.client('s3', region_name=REGION)
    s3.download_file(bucket, key, local_filename)

def wait_for_s3_file(s3_uri, timeout=3600, interval=300):
    """
    Waits for a file to become available in S3.

    :param s3_uri: The S3 URI of the file to wait for.
    :param timeout: Maximum time to wait in seconds (default: 1 hour).
    :param interval: Time between checks in seconds (default: 5 minutes).
    :raises TimeoutError: If the file is not available within the timeout period.
    """
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip('/')
    s3 = boto3.client('s3', region_name=REGION)
    
    start_time = time.time()
    while True:
        try:
            s3.head_object(Bucket=bucket, Key=key)
            print(f"File {s3_uri} is now available.")
            break
        except boto3.exceptions.botocore.client.ClientError as e:
            if e.response['Error']['Code'] == '404':
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    raise TimeoutError(f"Timeout: {s3_uri} not available after {timeout} seconds.")
                print(f"File {s3_uri} not found. Waiting for {interval} seconds before retrying...")
                time.sleep(interval)
            else:
                raise
