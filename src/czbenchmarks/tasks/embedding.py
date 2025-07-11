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
    "EmbeddingTaskInput",
    "EmbeddingMetricInput",
    "EmbeddingTask",
]


class EmbeddingTaskInput(TaskInput):
    """Pydantic model for EmbeddingTask inputs."""

    pass


class EmbeddingMetricInput(MetricInput):
    """Pydantic model for EmbeddingTask metric inputs."""

    input_labels: ListLike


class EmbeddingTask(BaseTask):
    """Task for evaluating cell representation quality using labeled data.

    This task computes quality metrics for cell representations using ground truth labels.
    Currently supports silhouette score evaluation.

    Args:
        random_seed (int): Random seed for reproducibility
    """

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.display_name = "embedding"

    def _run_task(
        self, cell_representation: CellRepresentation, task_input: EmbeddingTaskInput
    ) -> dict:
        return {"cell_representation": cell_representation}

    def _compute_metrics(
        self, task_output: dict, metric_input: EmbeddingMetricInput
    ) -> List[MetricResult]:
        """Computes cell representation quality metrics.

        Returns:
            List of MetricResult objects containing silhouette score
        """
        metric_type = MetricType.SILHOUETTE_SCORE
        cell_representation = task_output["cell_representation"]
        return [
            MetricResult(
                metric_type=metric_type,
                value=metrics_registry.compute(
                    metric_type,
                    X=cell_representation,
                    labels=metric_input.input_labels,
                ),
            )
        ]
