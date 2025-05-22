import logging
from typing import Set, List

from .constants import RANDOM_SEED
from ..datasets import BaseDataset, DataType
from ..models.types import ModelType
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from .base import BaseTask

logger = logging.getLogger(__name__)


class BatchIntegrationTask(BaseTask):
    """Task for evaluating batch integration quality.

    This task computes metrics to assess how well different batches are integrated
    in the embedding space while preserving biological signals.

    Args:
        label_key: Key to access ground truth cell type labels in metadata
        batch_key: Key to access batch labels in metadata
        random_seed (int): Random seed for reproducibility
    """

    def __init__(
        self, label_key: str, batch_key: str, *, random_seed: int = RANDOM_SEED
    ):
        super().__init__(random_seed=random_seed)
        self.label_key = label_key
        self.batch_key = batch_key

    @property
    def display_name(self) -> str:
        """A pretty name to use when displaying task results"""
        return "batch integration"

    @property
    def required_inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set of required input DataTypes (metadata with labels)
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
        """Runs the batch integration evaluation task.

        Gets embedding coordinates, batch labels and cell type labels from the dataset
        for metric computation.

        Args:
            data: Dataset containing embedding and labels
        """
        self.embedding = data.get_output(model_type, DataType.EMBEDDING)
        self.batch_labels = data.get_input(DataType.METADATA)[self.batch_key]
        self.labels = data.get_input(DataType.METADATA)[self.label_key]

    def _compute_metrics(self) -> List[MetricResult]:
        """Computes batch integration quality metrics.

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
                    X=self.embedding,
                    labels=self.batch_labels,
                ),
            ),
            MetricResult(
                metric_type=silhouette_batch_metric,
                value=metrics_registry.compute(
                    silhouette_batch_metric,
                    X=self.embedding,
                    labels=self.labels,
                    batch=self.batch_labels,
                ),
            ),
        ]
