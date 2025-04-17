from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Union

from ..datasets import BaseDataset, DataType
from ..models.types import ModelType
from ..metrics.types import MetricResult
from .utils import run_standard_scrna_workflow


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
    # pragma: no cover
    def required_inputs(self) -> Set[DataType]:
        """Required input data types this task requires.

        Returns:
            Set of DataType enums that must be present in input data
        """

    @property
    @abstractmethod
    # pragma: no cover
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

        # Check if there are any model outputs at all
        if not data.outputs:
            error_msg.append("No model outputs available")
        else:
            for model_type in data.outputs:
                missing_outputs = self.required_outputs - set(
                    data.outputs[model_type].keys()
                )
                if missing_outputs:
                    error_msg.append(
                        "Missing required outputs for model type "
                        f"{model_type.name}: {missing_outputs}"
                    )

        if error_msg:
            raise ValueError(
                f"Data validation failed for {self.__class__.__name__}: "
                f"{' | '.join(error_msg)}"
            )

        data.validate()

    @abstractmethod
    # pragma: no cover
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
    # pragma: no cover
    def _compute_metrics(self) -> List[MetricResult]:
        """Compute evaluation metrics for the task.

        Returns:
            List of MetricResult objects containing metric values and metadata
        """

    def _run_task_for_dataset(
        self,
        data: Union[BaseDataset, List[BaseDataset]],
        model_types: Optional[List[ModelType]] = None,
    ) -> Dict[ModelType, List[MetricResult]]:
        """Run task for a dataset or list of datasets and compute metrics for each model.

        This method determines which model types to evaluate, validates their
        availability, runs the task implementation for each model type, and
        computes the corresponding metrics.

        Args:
            data: Single dataset or list of datasets containing required input and
                  output data
            model_types: Optional list of specific model types to evaluate. If None,
                         will use all available model types common across datasets.

        Returns:
            Dictionary mapping model types to their list of MetricResult objects

        Raises:
            ValueError: If no common model types found across datasets or
                       if a specified model type is not available in a dataset
        """
        # Dictionary to store metrics for each model type
        all_metrics_per_model = {}

        # Determine which model types to evaluate if not explicitly provided
        if model_types is None:
            if isinstance(data, list):
                # For multiple datasets, find model types available in all datasets
                model_types = set.intersection(
                    *[set(dataset.outputs.keys()) for dataset in data]
                )
                if not model_types:
                    raise ValueError("No common model types found across all datasets")
            else:
                # For single dataset, use all available model types
                model_types = list(data.outputs.keys())

        # Validate that all requested model types are available in all datasets
        for model_type in model_types:
            if isinstance(data, list):
                for dataset in data:
                    if model_type not in dataset.outputs:
                        raise ValueError(
                            f"Model type {model_type} not found in dataset"
                        )
            else:
                if model_type not in data.outputs:
                    raise ValueError(f"Model type {model_type} not found in dataset")

        # Process each model type
        for model_type in model_types:
            # Run the task implementation for this model
            self._run_task(data, model_type)

            # Compute metrics based on task results
            metrics = self._compute_metrics()

            # Store metrics for this model type
            all_metrics_per_model[model_type] = metrics

        return all_metrics_per_model

    def set_baseline(self, data: BaseDataset, **kwargs):
        """Set a baseline embedding using PCA on gene expression data.

        This method performs standard preprocessing on the raw gene expression data
        and uses PCA for dimensionality reduction. It then sets the PCA embedding
        as the BASELINE model output in the dataset, which can be used for comparison
        with other model embeddings.

        Args:
            data: BaseDataset containing AnnData with gene expression data
            **kwargs: Additional arguments passed to run_standard_scrna_workflow
        """

        # Get the AnnData object from the dataset
        adata = data.get_input(DataType.ANNDATA)

        # Run the standard preprocessing workflow
        adata_baseline = run_standard_scrna_workflow(adata, **kwargs)

        # Use PCA result as the embedding for clustering
        data.set_output(
            ModelType.BASELINE, DataType.EMBEDDING, adata_baseline.obsm["X_pca"]
        )

    def run(
        self,
        data: Union[BaseDataset, List[BaseDataset]],
        model_types: Optional[List[ModelType]] = None,
    ) -> Union[
        Dict[ModelType, List[MetricResult]],
        List[Dict[ModelType, List[MetricResult]]],
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
