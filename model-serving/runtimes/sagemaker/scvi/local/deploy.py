import logging
from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorchModel


ROLE = "OmarSageMakerRole"

# If your local model artifact is stored locally, use file:// prefix.
LOCAL_MODEL_ARTIFACT = "file://scvi_model_code.tar.gz"
MODEL_NAME = "scvi-local"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def serve_model_locally():
    """
    Serve the model locally using SageMaker Local Mode.
    """
    # set_credentials()
    sagemaker_session = LocalSession()
    sagemaker_session.config = {'local': {'local_code': True}}

    # Create the PyTorch model for local mode.
    env_config = {
        'TS_MAX_RESPONSE_SIZE': '2147483647', # 2GB
        'TS_MAX_REQUEST_SIZE': '2147483647',
        # NOTE: The timeout hasn't been an issue in testing -
        # We've only tested files of size up to 2.4GB. But the timeout
        # might be an issue for larger files or larger models
        #'SAGEMAKER_MODEL_SERVER_TIMEOUT':'3600'
    }
    pytorch_model = PyTorchModel(
        model_data=LOCAL_MODEL_ARTIFACT,
        role=ROLE,
        framework_version="2.5",
        py_version="py311",
        entry_point="inference.py",
        source_dir="code/",
        sagemaker_session=sagemaker_session,
        name=MODEL_NAME,
        env=env_config)
    logger.info(f"Local model '{MODEL_NAME}' has been created.")

    # Deploy the model locally.
    # Use instance_type="local" (or "local_gpu" if GPU support is available).
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type="local_gpu",
        )
    logger.info("Model is deployed locally and ready to accept predictions.")

    return predictor

if __name__ == "__main__":
    serve_model_locally()
