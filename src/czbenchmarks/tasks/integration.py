import logging
from typing import List

from ..constants import RANDOM_SEED
from ..datasets.types import Embedding, ListLike
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from .base import BaseTask

logger = logging.getLogger(__name__)


class BatchIntegrationTask(BaseTask):
    """Task for evaluating batch integration quality.

    This task computes metrics to assess how well different batches are integrated
    in the embedding space while preserving biological signals.

    Args:
        random_seed (int): Random seed for reproducibility
    """

    def __init__(
        self, *, random_seed: int = RANDOM_SEED
    ):
        super().__init__(random_seed=random_seed)
        self.display_name = "batch integration"

    def _run_task(self, **kwargs) -> dict:
        return {}

    def _compute_metrics(
        self, embedding: Embedding, batch_labels: ListLike, labels: ListLike
    ) -> List[MetricResult]:
        """Computes batch integration quality metrics.

        Args:
            embedding: embedding to use for the task
            batch_labels: batch labels to use for the task
            labels: cell type labels to use for the task

        Returns:
            List of MetricResult objects containing entropy per cell and
            batch-aware silhouette scores
        """

        entropy_per_cell_metric = MetricType.ENTROPY_PER_CELL
        silhouette_batch_metric = MetricType.BATCH_SILHOUETTE

        return [
            MetricResult(
                metric_type=entropy_per_cell_metric,
                value=metrics_registry.compute(
                    entropy_per_cell_metric,
                    X=embedding,
                    labels=batch_labels,
                    random_seed=self.random_seed,
                ),
            ),
            MetricResult(
                metric_type=silhouette_batch_metric,
                value=metrics_registry.compute(
                    silhouette_batch_metric,
                    X=embedding,
                    labels=labels,
                    batch=batch_labels,
                ),
            ),
        ]
