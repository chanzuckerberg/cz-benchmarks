import logging
from typing import List

from ..constants import RANDOM_SEED
from ..datasets.types import CellRepresentation, ListLike
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from .base import BaseTask
from .types import TaskInput, MetricInput

logger = logging.getLogger(__name__)


__all__ = [
    "BatchIntegrationTaskInput",
    "BatchIntegrationMetricInput",
    "BatchIntegrationTask",
]


class BatchIntegrationTaskInput(TaskInput):
    """Pydantic model for BatchIntegrationTask inputs."""

    pass


class BatchIntegrationMetricInput(MetricInput):
    """Pydantic model for BatchIntegrationTask metric inputs."""

    batch_labels: ListLike
    labels: ListLike


class BatchIntegrationTask(BaseTask):
    """Task for evaluating batch integration quality.

    This task computes metrics to assess how well different batches are integrated
    in the embedding space while preserving biological signals.

    Args:
        random_seed (int): Random seed for reproducibility
    """

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.display_name = "batch integration"

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: BatchIntegrationTaskInput,
    ) -> dict:
        return {"cell_representation": cell_representation}

    def _compute_metrics(
        self, task_output: dict, metric_input: BatchIntegrationMetricInput
    ) -> List[MetricResult]:
        """Computes batch integration quality metrics.

        Args:
            cell_representation: gene expression data or embedding for task
            batch_labels: batch labels to use for the task
            labels: cell type labels to use for the task

        Returns:
            List of MetricResult objects containing entropy per cell and
            batch-aware silhouette scores
        """

        entropy_per_cell_metric = MetricType.ENTROPY_PER_CELL
        silhouette_batch_metric = MetricType.BATCH_SILHOUETTE
        cell_representation = task_output["cell_representation"]

        return [
            MetricResult(
                metric_type=entropy_per_cell_metric,
                value=metrics_registry.compute(
                    entropy_per_cell_metric,
                    X=cell_representation,
                    labels=metric_input.batch_labels,
                    random_seed=self.random_seed,
                ),
            ),
            MetricResult(
                metric_type=silhouette_batch_metric,
                value=metrics_registry.compute(
                    silhouette_batch_metric,
                    X=cell_representation,
                    labels=metric_input.labels,
                    batch=metric_input.batch_labels,
                ),
            ),
        ]
