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
    pytorch_model = PyTorchModel(
        model_data=LOCAL_MODEL_ARTIFACT,
        role=ROLE,
        framework_version="2.5",
        py_version="py311",
        entry_point="inference.py",
        source_dir="code/",
        sagemaker_session=sagemaker_session,
        name=MODEL_NAME)
    logger.info(f"Local model '{MODEL_NAME}' has been created.")

    # Deploy the model locally.
    # Use instance_type="local" (or "local_gpu" if GPU support is available).
    env_config = {
        "TS_MAX_RESPONSE_SIZE": "2000000000",
        "TS_MAX_REQUEST_SIZE": "2000000000"
    }
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type="local_gpu",
        env=env_config
        )
    logger.info("Model is deployed locally and ready to accept predictions.")

    return predictor

if __name__ == "__main__":
    serve_model_locally()
