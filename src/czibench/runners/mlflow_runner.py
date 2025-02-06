import json
import os
import mlflow
import requests

from .model_runner import ModelRunnerBase
from ..datasets.base import BaseDataset

class MLflowModelRunner(ModelRunnerBase):
    """Handles model execution logic for an MLflow model"""
    
    def run_local(self, dataset: BaseDataset) -> BaseDataset:
        dataset.output_embedding = mlflow.models.predict(
            model_uri=self.model_resource_url, input_data=dataset.path, env_manager="uv"
        )
    
        return dataset

    # TODO: test & debug!
    def run_remote(self, dataset: BaseDataset) -> BaseDataset:
        token = os.environ.get("DATABRICKS_TOKEN")
        if not token:
            raise EnvironmentError("DATABRICKS_TOKEN environment variable is missing")
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

        input_data = json.dumps({"inputs": [[dataset.path]]})

        response = requests.request(method='POST', headers=headers, url=self.model_endpoint, data=input_data)
        response.raise_for_status()
        
        dataset.output_embedding = response.json()
        
        return dataset
