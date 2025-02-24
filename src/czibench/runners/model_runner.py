from abc import abstractmethod
from typing import Any, Optional

from ..datasets.base import BaseDataset

class ModelRunnerBase:
    """Handles model execution logic for both local and remote models"""
    
    def __init__(
        self,
        model_resource_url: Optional[str] = None,
        model_endpoint: Optional[str] = None,
        **kwargs: Any
    ):

        if not (bool(model_resource_url) ^ bool(model_endpoint)):
            raise ValueError("Exactly one of `model_resource_url` or `model_endpoint_url` must be specified.")
        
        self.model_resource_url = model_resource_url
        self.model_endpoint = model_endpoint
        self.model_params = kwargs
        
    def run(self, dataset: BaseDataset) -> BaseDataset:
        """Run the model locally or remotely, depending on the ModelRunner's configuration."""
        if self.model_endpoint:
            dataset = self._run_remote(dataset)
        else:
            assert self.model_resource_url
            dataset = self._run_local(dataset)
                    
        return dataset

    @abstractmethod
    def _run_local(self, data: BaseDataset) -> BaseDataset:
         raise NotImplementedError("ModelRunner is an abstract class.")

    @abstractmethod
    def _run_remote(self, data: BaseDataset) -> BaseDataset:
         raise NotImplementedError("ModelRunner is an abstract class.")


