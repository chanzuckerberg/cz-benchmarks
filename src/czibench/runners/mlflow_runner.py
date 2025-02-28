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
            with tempfile.NamedTemporaryFile(mode="w") as input:
                input_json = json.dumps({"inputs": [[dataset.source_path]],
                                         "params": {"organism": str(dataset.get_input(DataType.ORGANISM))}})
                input.write(input_json)
                input.flush()
                print(f"Calling MLflow model process (localhost): {self.model_endpoint}")
                prediction = mlflow.models.predict(
                    model_uri=self.model_resource_url, 
                    # NOTE: You cannot pass params if you use `input_data`; must use `input_path` with a file containing JSON with `input_data` and `params` objects.
                    # input_data=dataset.local_path,
                    input_path=input.name,
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
                "params": {"organism": str(dataset.get_input(DataType.ORGANISM))},
            }
        )
        print(f"Calling MLflow model endpoint: {self.model_endpoint}")
        response = requests.request(method='POST', headers=headers, url=self.model_endpoint, data=input_data)
        response.raise_for_status()
        
        prediction = np.array(response.json()['predictions'])
        dataset.set_output(DataType.EMBEDDING, prediction)
        
        return dataset
