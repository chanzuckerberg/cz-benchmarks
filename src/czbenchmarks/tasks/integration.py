import logging
from typing import Annotated, List
from pydantic import Field, field_validator

import scipy.sparse as sp

from ..constants import RANDOM_SEED
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from ..tasks.types import CellRepresentation
from ..types import ListLike
from .task import PCABaselineInput, Task, TaskInput, TaskOutput

logger = logging.getLogger(__name__)


class BatchIntegrationTaskInput(TaskInput):
    """Pydantic model for BatchIntegrationTask inputs."""

    batch_labels: Annotated[
        ListLike, Field(description="Batch labels for each cell  (e.g. `obs.batch` from an AnnData object).")
    ]

    labels: Annotated[
        ListLike,
        Field(
            description="Ground truth labels for metric calculation (e.g. `obs.cell_type` from an AnnData object)."
        ),
    ]

    @field_validator("batch_labels")
    @classmethod
    def _validate_batch_labels(cls, v: ListLike) -> ListLike:
        if not isinstance(v, ListLike):
            raise ValueError("batch_labels must be a list-like object.")
        return v

    @field_validator("labels")
    @classmethod
    def _validate_labels(cls, v: ListLike) -> ListLike:
        if not isinstance(v, ListLike):
            raise ValueError("labels must be a list-like object.")
        return v


class BatchIntegrationOutput(TaskOutput):
    """Output for batch integration task."""

    cell_representation: CellRepresentation  # The cell representation matrix


class BatchIntegrationTask(Task):
    """Task for evaluating batch integration quality.

    This task computes metrics to assess how well different batches are integrated
    in the embedding space while preserving biological signals.
    """

    display_name = "Batch Integration"
    description = (
        "Evaluate batch integration quality using various integration metrics."
    )
    input_model = BatchIntegrationTaskInput
    baseline_model = PCABaselineInput

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        _: BatchIntegrationTaskInput,
    ) -> BatchIntegrationOutput:
        """Run the task's core computation.

        Args:
            cell_representation: gene expression data or embedding for task
            _: (unused) Pydantic model with inputs for the task
        Returns:
            BatchIntegrationOutput: Pydantic model with cell representation
        """
        return BatchIntegrationOutput(cell_representation=cell_representation)

    def _compute_metrics(
        self,
        task_input: BatchIntegrationTaskInput,
        task_output: BatchIntegrationOutput,
    ) -> List[MetricResult]:
        """Computes batch integration quality metrics.

        Args:
            task_input: Pydantic model with inputs for the task
            task_output: Pydantic model with task outputs from _run_task

        Returns:
            List of MetricResult objects containing entropy per cell and
            batch-aware silhouette scores
        """
        logger.debug("BatchIntegrationTask._compute_metrics: Computing metrics")
        entropy_per_cell_metric = MetricType.ENTROPY_PER_CELL
        silhouette_batch_metric = MetricType.BATCH_SILHOUETTE
        cell_representation = task_output.cell_representation
        logger.debug(
            f"BatchIntegrationTask._compute_metrics: cell_representation shape={cell_representation.shape}, labels shape={task_input.labels.shape}"
        )
        logger.debug(
            f"BatchIntegrationTask._compute_metrics: batch_labels shape={task_input.batch_labels.shape}"
        )

        # Convert sparse matrix to dense if needed for JAX compatibility in metrics
        if sp.issparse(cell_representation):
            cell_representation = cell_representation.toarray()

        return [
            MetricResult(
                metric_type=entropy_per_cell_metric,
                value=metrics_registry.compute(
                    entropy_per_cell_metric,
                    X=cell_representation,
                    labels=task_input.batch_labels,
                    random_seed=self.random_seed,
                ),
            ),
            MetricResult(
                metric_type=silhouette_batch_metric,
                value=metrics_registry.compute(
                    silhouette_batch_metric,
                    X=cell_representation,
                    labels=task_input.labels,
                    batch=task_input.batch_labels,
                ),
            ),
        ]
