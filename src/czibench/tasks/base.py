from abc import ABC, abstractmethod
from typing import Dict, Set
from ..datasets.base import BaseDataset
from ..datasets.types import DataType


class BaseTask(ABC):
    """Base class for all tasks"""

    @property
    @abstractmethod
    def required_inputs(self) -> Set[DataType]:
        """Specify what input types this task requires"""

    @property
    @abstractmethod
    def required_outputs(self) -> Set[DataType]:
        """Specify what output types from models this task requires"""

    def validate(self, data: BaseDataset):
        # Check both inputs and outputs are available
        missing_inputs = self.required_inputs - set(data.inputs.keys())
        missing_outputs = self.required_outputs - set(data.outputs.keys())

        if missing_inputs or missing_outputs:
            error_msg = []
            if missing_inputs:
                error_msg.append(f"Missing required inputs: {missing_inputs}")
            if missing_outputs:
                error_msg.append(f"Missing required outputs: {missing_outputs}")
            raise ValueError(
                f"Data validation failed for {self.__class__.__name__}: "
                f"{' | '.join(error_msg)}"
            )

        data.validate()

    @abstractmethod
    def _run_task(self, data: BaseDataset) -> BaseDataset:
        pass

    @abstractmethod
    def _compute_metrics(self) -> Dict[str, float]:
        pass

    def run(self, data: BaseDataset) -> BaseDataset:
        self.validate(data)

        data = self._run_task(data)
        results = self._compute_metrics()
        return data, results
