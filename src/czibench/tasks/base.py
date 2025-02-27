from abc import ABC, abstractmethod
from typing import Dict, Set, Union, List
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

    @property
    def requires_multiple_datasets(self) -> bool:
        """Whether this task requires multiple datasets"""
        return False

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
    def _run_task(self, data: Union[BaseDataset, List[BaseDataset]]):
        pass

    @abstractmethod
    def _compute_metrics(self) -> Dict[str, float]:
        pass

    def run(
        self, data: Union[BaseDataset, List[BaseDataset]]
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        if isinstance(data, BaseDataset):
            self.validate(data)
        elif isinstance(data, list) and all(isinstance(d, BaseDataset) for d in data):
            for d in data:
                self.validate(d)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")

        if self.requires_multiple_datasets and not isinstance(data, list):
            raise ValueError("This task requires a list of datasets")

        if isinstance(data, list) and not self.requires_multiple_datasets:
            all_metrics = []
            for d in data:
                self._run_task(d)
                metrics = self._compute_metrics()
                all_metrics.append(metrics)
            return all_metrics
        else:
            self._run_task(data)
            return self._compute_metrics()
