from abc import ABC, abstractmethod
from typing import Dict, List, Set, Union

from ..datasets.base import BaseDataset
from ..datasets.types import DataType


class BaseTask(ABC):
    """Abstract base class for all benchmark tasks.

    Defines the interface that all tasks must implement. Tasks are responsible for:
    1. Declaring their required input/output data types
    2. Running task-specific computations
    3. Computing evaluation metrics

    Tasks should store any intermediate results as instance variables
    to be used in metric computation.
    """

    @property
    @abstractmethod
    def required_inputs(self) -> Set[DataType]:
        """Required input data types this task requires.

        Returns:
            Set of DataType enums that must be present in input data
        """

    @property
    @abstractmethod
    def required_outputs(self) -> Set[DataType]:
        """Required output types from models this task requires

        Returns:
            Set of DataType enums that must be present in output data
        """

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
    def _run_task(self, data: BaseDataset) -> BaseDataset:
        """Run the task's core computation.

        Should store any intermediate results needed for metric computation
        as instance variables.

        Args:
            data: Dataset containing required input and output data

        Returns:
            Modified or unmodified dataset
        """

    @abstractmethod
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute evaluation metrics for the task.

        Returns:
            Dictionary mapping metric names to their float values
        """

    def run(
        self, data: Union[BaseDataset, List[BaseDataset]]
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Run the task on input data and compute metrics.

        Args:
            data: Single dataset or list of datasets to evaluate. Must contain
                required input and output data types.

        Returns:
            For single dataset: Dictionary of metric name to value
            For multiple datasets: List of metric dictionaries, one per dataset

        Raises:
            ValueError: If data is invalid type or missing required fields
            ValueError: If task requires multiple datasets but single dataset provided
        """
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
