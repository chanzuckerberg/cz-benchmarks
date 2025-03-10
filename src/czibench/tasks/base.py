from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Union

from ..datasets import BaseDataset
from ..datasets import DataType
from ..models.types import ModelType
from ..metrics.types import MetricType


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
        error_msg = []
        missing_inputs = self.required_inputs - set(data.inputs.keys())

        if missing_inputs:
            error_msg.append(f"Missing required inputs: {missing_inputs}")

        for model_type in data.outputs:
            missing_outputs = self.required_outputs - set(
                data.outputs[model_type].keys()
            )
            if missing_outputs:
                error_msg.append(
                    "Missing required outputs for model type"
                    f"{model_type.name}: {missing_outputs}"
                )

        if error_msg:
            raise ValueError(
                f"Data validation failed for {self.__class__.__name__}: "
                f"{' | '.join(error_msg)}"
            )

        data.validate()

    @abstractmethod
    def _run_task(self, data: BaseDataset, model_type: ModelType):
        """Run the task's core computation.

        Should store any intermediate results needed for metric computation
        as instance variables.

        Args:
            data: Dataset containing required input and output data

        Returns:
            Modified or unmodified dataset
        """

    @abstractmethod
    def _compute_metrics(self) -> Dict[MetricType, float]:
        """Compute evaluation metrics for the task.

        Returns:
            Dictionary mapping metric names to their float values
        """

    def _run_task_for_dataset(
        self, data: BaseDataset, model_types: Optional[List[ModelType]] = None
    ) -> Dict[ModelType, Dict[MetricType, float]]:
        """Run task for a single dataset and compute metrics for each model.

        This method iterates through all model outputs in the dataset,
        runs the task for each model, and computes the corresponding metrics.

        Args:
            data: Dataset containing required input and output data

        Returns:
            Dictionary mapping model types to their metric results
        """
        # Store metrics for each model in the dataset
        all_metrics_per_model = {}

        if model_types is None:
            model_types = list(data.outputs.keys())

        # Iterate through each model type in the dataset outputs
        for model_type in model_types:
            # Run the task implementation for this model
            self._run_task(data, model_type)

            # Compute metrics based on task results
            metrics = self._compute_metrics()

            # Convert numpy floats to python floats where possible
            for metric_name, metric_value in metrics.items():
                try:
                    metrics[metric_name] = float(metric_value)
                except ValueError:
                    pass

            # Store metrics for this model
            all_metrics_per_model[model_type] = metrics

        return all_metrics_per_model  # Return metrics for all models

    def run(
        self,
        data: Union[BaseDataset, List[BaseDataset]],
        model_types: Optional[List[ModelType]] = None,
    ) -> Union[
        Dict[ModelType, Dict[MetricType, float]],
        List[Dict[ModelType, Dict[MetricType, float]]],
    ]:
        """Run the task on input data and compute metrics.

        Args:
            data: Single dataset or list of datasets to evaluate. Must contain
                required input and output data types.

        Returns:
            For single dataset: Dictionary of model types to metric results
            For multiple datasets: List of metric dictionaries, one per dataset

        Raises:
            ValueError: If data is invalid type or missing required fields
            ValueError: If task requires multiple datasets but single dataset provided
        """
        # Validate input data type and required fields
        if isinstance(data, BaseDataset):
            self.validate(data)
        elif isinstance(data, list) and all(isinstance(d, BaseDataset) for d in data):
            for d in data:
                self.validate(d)
        else:
            raise ValueError(f"Invalid data type: {type(data)}")

        # Check if task requires multiple datasets
        if self.requires_multiple_datasets and not isinstance(data, list):
            raise ValueError("This task requires a list of datasets")

        # Handle single vs multiple datasets
        if isinstance(data, list) and not self.requires_multiple_datasets:
            # Process each dataset individually
            all_metrics = []
            for d in data:
                all_metrics.append(self._run_task_for_dataset(d, model_types))
            return all_metrics
        else:
            # Process single dataset or multiple datasets as required by the task
            return self._run_task_for_dataset(data, model_types)
