import subprocess
import time

import bentoml
import numpy as np

from ..datasets.base import BaseDataset
from ..datasets.types import DataType
from .model_runner import ModelRunnerBase

DEFAULT_LOCAL_ENDPOINT = "http://localhost:3000"

class BentomlModelRunner(ModelRunnerBase):
    """Handles model execution logic for an MLflow model"""

    def _run_forward_pass(self, dataset: BaseDataset) -> BaseDataset:
        client = bentoml.SyncHTTPClient(self.model_endpoint or DEFAULT_LOCAL_ENDPOINT)
        task = client.predict.submit(file_path=dataset.source_path)

        while task.get_status().value != "success":
            continue

        result = task.get()

        result = np.array(result["embedding"])
        dataset.set_output(DataType.EMBEDDING, result)
        return dataset

    def _run_local(self, dataset: BaseDataset) -> BaseDataset:
        proc = subprocess.Popen(
            ["bentoml", "serve", self.model_resource_url, "--port", "3000"]
        )

        time.sleep(2)  # wait a bit for server to start listening

        print("Bentoml server listening on http://localhost:3000")

        try:
            dataset = self._run_forward_pass(dataset)
            return dataset
        finally:
            # kill the server
            proc.terminate()
            proc.wait()

    def _run_remote(self, dataset: BaseDataset) -> BaseDataset:
        return self._run_forward_pass(dataset)
