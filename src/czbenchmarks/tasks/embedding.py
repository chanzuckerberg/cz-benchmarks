import logging
from typing import List

from ..constants import RANDOM_SEED
from ..datasets.types import Embedding, ListLike
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from .base import BaseTask

logger = logging.getLogger(__name__)


class EmbeddingTask(BaseTask):
    """Task for evaluating embedding quality using labeled data.

    This task computes quality metrics for embeddings using ground truth labels.
    Currently supports silhouette score evaluation.

    Args:
        label_key (str): Key to access ground truth labels in metadata
        random_seed (int): Random seed for reproducibility
    """

    def __init__(self, label_key: str, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.display_name = "embedding"
        self.label_key = label_key

    def _run_task(self, **kwargs):
        return {}

    def _compute_metrics(
        self, embedding: Embedding, input_labels: ListLike
    ) -> List[MetricResult]:
        """Computes embedding quality metrics.

        Returns:
            List of MetricResult objects containing silhouette score
        """
        metric_type = MetricType.SILHOUETTE_SCORE
        return [
            MetricResult(
                metric_type=metric_type,
                value=metrics_registry.compute(
                    metric_type,
                    X=embedding,
                    labels=input_labels,
                ),
            )
        ]
