import logging
from typing import List

from ..constants import RANDOM_SEED
from ..datasets.types import CellRepresentation, ListLike
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from .base import BaseTask

logger = logging.getLogger(__name__)


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

    def _run_task(self, cell_representation: CellRepresentation, **kwargs):
        return {}

    def _compute_metrics(
        self, cell_representation: CellRepresentation, input_labels: ListLike
    ) -> List[MetricResult]:
        """Computes cell representation quality metrics.

        Returns:
            List of MetricResult objects containing silhouette score
        """
        metric_type = MetricType.SILHOUETTE_SCORE
        return [
            MetricResult(
                metric_type=metric_type,
                value=metrics_registry.compute(
                    metric_type,
                    X=cell_representation,
                    labels=input_labels,
                ),
            )
        ]
