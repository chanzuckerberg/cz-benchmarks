from abc import ABC, abstractmethod
from typing import List, Union
import anndata as ad
from pydantic import BaseModel

from ..constants import RANDOM_SEED
from .types import CellRepresentation
from ..metrics.types import MetricResult
from .utils import run_standard_scrna_workflow


class TaskInput(BaseModel):
    """Base class for task inputs."""

    model_config = {"arbitrary_types_allowed": True}


class MetricInput(BaseModel):
    """Base class for metric inputs."""

    model_config = {"arbitrary_types_allowed": True}


class TaskOutput(BaseModel):
    """Base class for task outputs."""

    model_config = {"arbitrary_types_allowed": True}


class Task(ABC):
    """Abstract base class for all benchmark tasks.

    Defines the interface that all tasks must implement. Tasks are responsible for:
    1. Declaring their required input/output data types
    2. Running task-specific computations
    3. Computing evaluation metrics

    Tasks should store any intermediate results as instance variables
    to be used in metric computation.

    Args:
        random_seed (int): Random seed for reproducibility
    """

    def __init__(
        self,
        *,
        random_seed: int = RANDOM_SEED,
    ):
        self.random_seed = random_seed
        # FIXME should this be changed to requires_multiple_embeddings?
        self.requires_multiple_datasets = False

    @abstractmethod
    def _run_task(
        self, cell_representation: CellRepresentation, task_input: TaskInput
    ) -> TaskOutput:
        """Run the task's core computation.

        Should store any intermediate results needed for metric computation
        as instance variables.

        Args:
            cell_representation: gene expression data or embedding for task
            task_input: Pydantic model with inputs for the task
        Returns:
            TaskOutput: Pydantic model with output data for the task
        """

    @abstractmethod
    def _compute_metrics(
        self, task_output: TaskOutput, metric_input: MetricInput
    ) -> List[MetricResult]:
        """Compute evaluation metrics for the task.

        Returns:
            List of MetricResult objects containing metric values and metadata
        """

    def _run_task_for_dataset(
        self,
        cell_representation: CellRepresentation,
        task_input: TaskInput,
        metric_input: MetricInput,
    ) -> List[MetricResult]:
        """Run task for a dataset or list of datasets and compute metrics.

        This method runs the task implementation and computes the corresponding metrics.

        Args:
            cell_representation: gene expression data or embedding for task
            task_input: Pydantic model with inputs for the task
            metric_input: Pydantic model with inputs for the metrics
        Returns:
            List of MetricResult objects

        """

        task_output = self._run_task(cell_representation, task_input)
        metrics = self._compute_metrics(task_output, metric_input)
        return metrics

    def compute_baseline(
        self,
        expression_data: CellRepresentation,
        **kwargs,
    ) -> CellRepresentation:
        """Set a baseline embedding using PCA on gene expression data.

        This method performs standard preprocessing on the raw gene expression data
        and uses PCA for dimensionality reduction. It then sets the PCA embedding
        as the BASELINE model output in the dataset, which can be used for comparison
        with other model embeddings.

        Args:
            expression_data: expression data to use for anndata
            **kwargs: Additional arguments passed to run_standard_scrna_workflow
        """

        # Create the AnnData object
        adata = ad.AnnData(X=expression_data)

        # Run the standard preprocessing workflow
        embedding_baseline = run_standard_scrna_workflow(adata, **kwargs)
        return embedding_baseline

    def run(
        self,
        cell_representation: Union[CellRepresentation, List[CellRepresentation]],
        task_input: TaskInput,
        metric_input: MetricInput,
    ) -> List[MetricResult]:
        """Run the task on input data and compute metrics.

        Args:
            cell_representation: gene expression data or embedding to use for the task
            task_input: Pydantic model with inputs for the task
            metric_input: Pydantic model with inputs for the metrics

        Returns:
            For single embedding: A one-element list containing a single metric result for the task
            For multiple embeddings: List of metric results for each task, one per dataset

        Raises:
            ValueError: If input does not match multiple embedding requirement
        """

        # Check if task requires embeddings from multiple datasets
        if self.requires_multiple_datasets:
            error_message = "This task requires a list of cell representations"
            if not isinstance(cell_representation, list):
                raise ValueError(error_message)
            if not all(
                [isinstance(emb, CellRepresentation) for emb in cell_representation]
            ):
                raise ValueError(error_message)
            if len(cell_representation) < 2:
                raise ValueError(f"{error_message} but only one was provided")
        else:
            if not isinstance(cell_representation, CellRepresentation):
                raise ValueError(
                    "This task requires a single cell representation for input"
                )

        return self._run_task_for_dataset(
            cell_representation,  # type: ignore
            task_input,
            metric_input,
        )
