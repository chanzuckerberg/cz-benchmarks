import bentoml
import numpy as np

from ..datasets.base import BaseDataset
from ..datasets.types import DataType
from .model_runner import ModelRunnerBase


class BentomlModelRunner(ModelRunnerBase):
    """Handles model execution logic for an MLflow model"""

    def _run_forward_pass(self, dataset: BaseDataset) -> BaseDataset:
        client = bentoml.SyncHTTPClient("http://localhost:3000")
        task = client.predict.submit(file_path=dataset.source_path)

        while task.get_status().value != "success":
            continue

        result = task.get()

        result = np.array(result["embedding"])
        dataset.set_output(DataType.EMBEDDING, result)
        return dataset

    def _run_local(self, dataset: BaseDataset) -> BaseDataset:
        return self._run_forward_pass(dataset)

    def _run_remote(self, dataset: BaseDataset) -> BaseDataset:
        return self._run_forward_pass(dataset)
