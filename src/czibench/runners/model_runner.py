from abc import abstractmethod
from typing import Any, Optional
import time

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
        start_time = time.perf_counter()
        
        
        if self.model_endpoint:
            dataset = self._run_remote(dataset)
        else:
            assert self.model_resource_url
            dataset = self._run_local(dataset)
                    
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Execution Time: {elapsed_time:.2f} seconds")
        
        return dataset

    @abstractmethod
    def _run_local(self, data: BaseDataset) -> BaseDataset:
         raise NotImplementedError("ModelRunner is an abstract class.")

    @abstractmethod
    def _run_remote(self, data: BaseDataset) -> BaseDataset:
         raise NotImplementedError("ModelRunner is an abstract class.")


