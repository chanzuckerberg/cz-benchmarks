import json
import os
import tempfile
import mlflow
import numpy as np
import requests

from ..datasets.types import DataType

from .model_runner import ModelRunnerBase
from ..datasets.base import BaseDataset

class MLflowModelRunner(ModelRunnerBase):
    """Handles model execution logic for an MLflow model"""

    def _run_local(self, dataset: BaseDataset) -> BaseDataset:
        with tempfile.NamedTemporaryFile(mode="w") as output:
            prediction = mlflow.models.predict(
                model_uri=self.model_resource_url, 
                input_data=dataset.local_path,
                # FIXME: Figure out how to pass additional params to model, if possible
                # params={"organism": dataset.get_input(DataType.ORGANISM)}, 
                output_path=output.name,
                env_manager="uv",
                # FIXME: this is not working; as is, it uses /tmp/
                extra_envs={"MLFLOW_ENV_ROOT": "mlflow-envs"}
            )
            output.flush()
            with open(output.name) as f:
                prediction_json = json.load(f)
                prediction = np.array(prediction_json["predictions"])
                dataset.set_output(DataType.EMBEDDING, prediction)
            return dataset

    def _run_remote(self, dataset: BaseDataset) -> BaseDataset:
        token = os.environ.get("DATABRICKS_TOKEN")
        if not token:
            raise EnvironmentError("DATABRICKS_TOKEN environment variable is missing")
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

        input_data = json.dumps(
            {
                "inputs": [[dataset.source_path]],
                # FIXME: Figure out how to pass additional params to model, if possible
                # "params": {"organism": str(dataset.get_input(DataType.ORGANISM))},
            }
        )

        response = requests.request(method='POST', headers=headers, url=self.model_endpoint, data=input_data)
        response.raise_for_status()
        
        prediction = np.array(response.json()['predictions'])
        dataset.set_output(DataType.EMBEDDING, prediction)
        
        return dataset
