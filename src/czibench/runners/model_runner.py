from abc import abstractmethod
from typing import Any

from ..datasets.base import BaseDataset

class ModelRunnerBase:
    """Handles model execution logic"""
    
    def __init__(
        self,
        model_url: str,
        gpu: bool = False,
        **kwargs: Any
    ):
        self.model_url = model_url
        self.gpu = gpu
        self.model_params = kwargs

    @abstractmethod
    def run(self, data: BaseDataset) -> BaseDataset:
         raise NotImplementedError("ModelRunner is an abstract class.")


