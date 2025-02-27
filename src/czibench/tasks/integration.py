from typing import Dict, Set
import logging
from .base import BaseTask
from ..datasets.single_cell import SingleCellDataset
from ..datasets.types import DataType
from .utils import compute_entropy_per_cell
from scib_metrics import silhouette_batch

logger = logging.getLogger(__name__)


class BatchIntegrationTask(BaseTask):
    """Task for evaluating batch integration quality.

    This task computes metrics to assess how well different batches are integrated
    in the embedding space while preserving biological signals.

    Args:
        label_key: Key to access ground truth cell type labels in metadata
        batch_key: Key to access batch labels in metadata
    """

    def __init__(self, label_key: str, batch_key: str):
        self.label_key = label_key
        self.batch_key = batch_key

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

    def _run_task(self, data: SingleCellDataset):
        """Runs the batch integration evaluation task.

        Gets embedding coordinates, batch labels and cell type labels from the dataset
        for metric computation.

        Args:
            data: Dataset containing embedding and labels
        """
        self.embedding = data.get_output(DataType.EMBEDDING)
        self.batch_labels = data.get_input(DataType.METADATA)[self.batch_key]
        self.labels = data.get_input(DataType.METADATA)[self.label_key]

    def _compute_metrics(self) -> Dict[str, float]:
        """Computes batch integration quality metrics.

        Returns:
            Dictionary containing entropy per cell and batch-aware silhouette scores
        """
        return {
            "entropy_per_cell": compute_entropy_per_cell(
                self.embedding, self.batch_labels
            ),
            "silhouette_score": silhouette_batch(
                self.embedding, self.labels, self.batch_labels
            ),
        }
