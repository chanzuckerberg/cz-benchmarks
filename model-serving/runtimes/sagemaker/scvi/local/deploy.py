import logging
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorchModel
from botocore import credentials

import os
import subprocess
# For local mode, the role is not really used. Dummy role:
DUMMY_ROLE = "OmarSageMakerRole"

# If your local model artifact is stored locally, use file:// prefix.
LOCAL_MODEL_ARTIFACT = "file://scvi_model_code.tar.gz"
MODEL_NAME = "scvi-local"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def set_credentials():
    """
    This function is only needed for local mode running on a Mac. Not needed for local mode running on a EC2 instance.
    Get AWS credentials by running the aws-oidc command locally and exporting the credentials to the environment.

    Sagemaker has an issue with trying to generate credentials with the aws-oidc command (under `credential_process` in ~/.aws/config) in the local mode.
    This is a workaround to set the credentials manually. Credentials are valid for 8 hrs.
    """
    aws_profile = os.getenv('AWS_PROFILE')
    if not aws_profile:
        logger.error("AWS_PROFILE environment variable is not set.")
        return None
    try:
        # Run the aws-oidc command to get credentials
        command = f"aws-oidc env --profile {aws_profile} "
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info("AWS credentials obtained successfully.")
        output = result.stdout

        # Split the output by newlines and set the environment variables
        for line in output.strip().split('\n'):
            if line:
                key, value = line.split('=', 1)
                os.environ[key] = value
                logger.info(f"Exported {key} to environment variables.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to obtain AWS credentials: {e.stderr}")
        return None

def create_local_pytorch_model(sm_session, role: str) -> PyTorchModel:
    """
    Create a PyTorch model for local serving using SageMaker Local Mode.
    """
    pytorch_model = PyTorchModel(
        model_data=LOCAL_MODEL_ARTIFACT,
        role=role,
        framework_version="2.5",
        py_version="py311",
        entry_point="inference.py",  # Your inference code
        source_dir="code/",          # Directory containing any additional code/dependencies
        sagemaker_session=sm_session,
        name=MODEL_NAME,
    )
    logger.info(f"Local model '{MODEL_NAME}' has been created.")
    return pytorch_model


def serve_model_locally():
    """
    Serve the model locally using SageMaker Local Mode.
    """
    set_credentials()
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # Create the PyTorch model for local mode.
    model = create_local_pytorch_model(sagemaker_session, DUMMY_ROLE)

    # Deploy the model locally.
    # Use instance_type="local" (or "local_gpu" if GPU support is available).
    predictor = model.deploy(initial_instance_count=1, instance_type="local")
    logger.info("Model is deployed locally and ready to accept predictions.")

    return predictor


if __name__ == "__main__":
    serve_model_locally()
