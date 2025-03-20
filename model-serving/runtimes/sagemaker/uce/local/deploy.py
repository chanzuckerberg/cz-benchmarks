from sagemaker.pytorch import PyTorchModel
from sagemaker.local import LocalSession
import sagemaker
import boto3
import logging
from omegaconf import OmegaConf
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODEL_NAME = "uce"

def serve_uce_model_locally():
    """
    Serve UCE locally using SageMaker Local Mode.
    """
    sagemaker_session = LocalSession(boto_session=boto3.Session(region_name="us-west-2"))
    sagemaker_session.config = {'local': {'local_code': True}}
    config = OmegaConf.load("config.yaml")
    
    pytorch_model = PyTorchModel(
        name=MODEL_NAME,
        role="OmarSageMakerRole",
        framework_version="2.1",
        py_version="py310",
        entry_point="inference.py",
        source_dir=".",
        sagemaker_session=sagemaker_session,
        git_config={
            "repo": "https://github.com/giovp/UCE.git",
            "branch": "sagemaker",  # Change if needed
            # "commit": "abc123",  # Optional: specify a commit
            # "token": "ghp_xxxx",  # Required only for private repos
        },
        model_data="s3://omar-data/uce/model.tar.gz",
    )

    logger.info(f"Local model '{MODEL_NAME}' has been created.")

    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type="local_gpu",
        )
    logger.info("Model is deployed locally and ready to accept predictions.")

    return predictor

if __name__ == "__main__":
    serve_uce_model_locally()
