from abc import ABC, abstractmethod
from typing import Dict, Set

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

    def validate(self, data: BaseDataset):
        # Check both inputs and outputs are available
        missing_inputs = self.required_inputs - set(data.inputs.keys())
        missing_outputs = self.required_outputs - set(data.outputs.keys())

        if missing_inputs or missing_outputs:
            error_msg = []
            if missing_inputs:
                error_msg.append(f"Missing required inputs: {missing_inputs}")
            if missing_outputs:
                error_msg.append(
                    f"Missing required outputs: {missing_outputs}"
                )
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

        Uses results stored during _run_task to compute metrics.

        Returns:
            Dictionary mapping metric names to values
        """

    def run(self, data: BaseDataset) -> Dict[str, float]:
        """Run the full task pipeline.

        1. Validates input/output data requirements
        2. Runs the task computation
        3. Computes evaluation metrics

        Args:
            data: Dataset containing required input and output data

        Returns:
            Dictionary containing computed metrics
        """
        self.validate(data)

        data = self._run_task(data)
        results = self._compute_metrics()
        return data, results
