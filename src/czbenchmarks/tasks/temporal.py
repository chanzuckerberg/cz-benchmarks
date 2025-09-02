import logging
from typing import List, Set

from ..datasets import BaseDataset, DataType
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from ..models.types import ModelType
from .base import BaseTask
from .constants import RANDOM_SEED

logger = logging.getLogger(__name__)


class TemporalTask(BaseTask):
    """Task for evaluating temporal consistency in embeddings.

    This task computes temporal quality metrics for embeddings using time point labels.
    Evaluates how well embeddings preserve temporal organization between cells.

    Args:
        time_key (str): Key to access time point labels in metadata
        k (int): Number of neighbors for k-NN based metrics (default: 15)
        random_seed (int): Random seed for reproducibility
    """

    def __init__(self, time_key: str, k: int = 15, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.time_key = time_key
        self.k = k

    @property
    def display_name(self) -> str:
        """A pretty name to use when displaying task results"""
        return "temporal"

    @property
    def required_inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set of required input DataTypes (metadata with time labels)
        """
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        """Required output data types.

        Returns:
            required output types from models this task to run (embedding coordinates)
        """
        return {DataType.EMBEDDING}

    def _run_task(self, data: BaseDataset, model_type: ModelType):
        """Runs the temporal evaluation task.

        Gets embedding coordinates and time labels from the dataset for metric computation.

        Args:
            data: Dataset containing embedding and time labels
        """
        # Store embedding and time labels for metric computation
        self.embedding = data.get_output(model_type, DataType.EMBEDDING)
        self.time_labels = data.get_input(DataType.METADATA)[self.time_key]

    def _compute_metrics(self) -> List[MetricResult]:
        """Computes temporal consistency metrics.

        Returns:
            List of MetricResult objects containing temporal metrics
        """
        results = []

        # Temporal Silhouette Score
        results.append(
            MetricResult(
                metric_type=MetricType.TEMPORAL_SILHOUETTE,
                value=metrics_registry.compute(
                    MetricType.TEMPORAL_SILHOUETTE,
                    X=self.embedding,
                    time_labels=self.time_labels,
                ),
                params={},
            )
        )

        # Temporal Smoothness Score
        results.append(
            MetricResult(
                metric_type=MetricType.TEMPORAL_SMOOTHNESS,
                value=metrics_registry.compute(
                    MetricType.TEMPORAL_SMOOTHNESS,
                    emb=self.embedding,
                    time_var=self.time_labels,
                    k=self.k,
                ),
                params={"k": self.k},
            )
        )

        return results
