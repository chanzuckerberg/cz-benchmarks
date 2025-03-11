import logging
from typing import Dict, Set, List

from ..datasets import BaseDataset, DataType
from ..models.types import ModelType
from ..metrics import MetricType, metrics
from ..metrics.types import MetricResult
from .base import BaseTask

logger = logging.getLogger(__name__)


class EmbeddingTask(BaseTask):
    """Task for evaluating embedding quality using labeled data.

    This task computes quality metrics for embeddings using ground truth labels.
    Currently supports silhouette score evaluation.

    Args:
        label_key (str): Key to access ground truth labels in metadata
    """

    def __init__(self, label_key: str):
        self.label_key = label_key

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
            required output types from models this task to run  (embedding coordinates)
        """
        return {DataType.EMBEDDING}

    def _run_task(self, data: BaseDataset, model_type: ModelType):
        """Runs the embedding evaluation task.

        Gets embedding coordinates and labels from the dataset for metric computation.

        Args:
            data: Dataset containing embedding and labels
        """
        # Store embedding and labels for metric computation
        self.embedding = data.get_output(model_type, DataType.EMBEDDING)
        self.input_labels = data.get_input(DataType.METADATA)[self.label_key]

    def _compute_metrics(self) -> List[MetricResult]:
        """Computes embedding quality metrics.

        Returns:
            List of MetricResult objects containing silhouette score
        """
        metric_type = MetricType.SILHOUETTE_SCORE
        return [
            MetricResult(
                metric_type=metric_type,
                value=metrics.compute(
                    metric_type,
                    X=self.embedding,
                    labels=self.input_labels,
                ),
            )
        ]
