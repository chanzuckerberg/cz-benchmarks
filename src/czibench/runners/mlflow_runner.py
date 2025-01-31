import mlflow

from .model_runner import ModelRunnerBase
from ..datasets.base import BaseDataset

class MLflowModelRunner(ModelRunnerBase):
    """Handles model execution logic for an MLflow model"""
    
    def run(self, data: BaseDataset) -> BaseDataset:
        data.output_embedding = mlflow.models.predict(
            model_uri=self.model_url, input_data=data.path, env_manager="uv"
        )
    
        return data
