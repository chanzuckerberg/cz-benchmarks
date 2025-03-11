import logging
from typing import Set, List
import scanpy as sc

from ..datasets import BaseDataset, DataType
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from ..models.types import ModelType
from .base import BaseTask
from .utils import cluster_embedding

logger = logging.getLogger(__name__)


class ClusteringTask(BaseTask):
    """Task for evaluating clustering performance against ground truth labels.

    This task performs clustering on embeddings and evaluates the results
    using multiple clustering metrics (ARI and NMI).

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
            required output types from models this task to run (embedding to cluster)
        """
        return {DataType.EMBEDDING}

    def _run_task(self, data: BaseDataset, model_type: ModelType):
        """Runs clustering on the embedding data.

        Performs clustering and stores results for metric computation.

        Args:
            data: Dataset containing embedding and ground truth labels
        """
        # Get anndata object and add embedding
        adata = data.adata
        adata.obsm["emb"] = data.get_output(model_type, DataType.EMBEDDING)

        # Store labels and generate clusters
        self.input_labels = data.get_input(DataType.METADATA)[self.label_key]
        self.predicted_labels = cluster_embedding(adata, obsm_key="emb")

    def _compute_metrics(self) -> List[MetricResult]:
        """Computes clustering evaluation metrics.

        Returns:
            List of MetricResult objects containing ARI and NMI scores
        """
        return [
            MetricResult(
                metric_type=metric_type,
                value=metrics_registry.compute(
                    metric_type,
                    labels_true=self.input_labels,
                    labels_pred=self.predicted_labels,
                ),
            )
            for metric_type in [
                MetricType.ADJUSTED_RAND_INDEX,
                MetricType.NORMALIZED_MUTUAL_INFO,
            ]
        ]
