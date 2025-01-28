from abc import ABC, abstractmethod
from typing import Dict
from ..datasets.base import BaseDataset


class BaseTask(ABC):
    """Base class for all tasks"""

    @abstractmethod
    def validate(self, data: BaseDataset) -> bool:
        pass

    @abstractmethod
    def _run_task(self, data: BaseDataset) -> BaseDataset:
        pass

    @abstractmethod
    def _compute_metrics(self) -> Dict[str, float]:
        pass

    def run(self, data: BaseDataset) -> BaseDataset:
        if not self.validate(data):
            raise ValueError(f"Data validation failed for {self.__class__.__name__}")
        data = self._run_task(data)
        results = self._compute_metrics()
        return data, results
