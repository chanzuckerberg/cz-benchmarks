from abc import abstractmethod
from operator import xor
from typing import Any

from ..datasets.base import BaseDataset

class ModelRunnerBase:
    """Handles model execution logic for both local and remote models"""
    
    def __init__(
        self,
        model_resource_url: str,
        model_endpoint: str,
        **kwargs: Any
    ):

        if not (bool(model_resource_url) ^ bool(model_endpoint)):
            raise ValueError("Exactly one of `model_resource_url` or `model_endpoint_url` must be specified.")
        
        self.model_resource_url = model_resource_url
        self.model_endpoint = model_endpoint
        self.model_params = kwargs
        
    def run(self, data: BaseDataset) -> BaseDataset:
        """Run the model locally or remotely, depending on the ModelRunner's configuration."""
        if self.model_endpoint:
            return self.run_remote(data)
        else:
            assert self.model_resource_url
            return self.run_local(data)

    @abstractmethod
    def run_local(self, data: BaseDataset) -> BaseDataset:
         raise NotImplementedError("ModelRunner is an abstract class.")

    @abstractmethod
    def run_remote(self, data: BaseDataset) -> BaseDataset:
         raise NotImplementedError("ModelRunner is an abstract class.")


