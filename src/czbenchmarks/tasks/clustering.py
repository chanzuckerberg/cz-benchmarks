import logging
from typing import List
import numpy as np

from ..datasets import BaseDataset
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from .base import BaseTask
from .utils import cluster_embedding
from .constants import RANDOM_SEED, N_ITERATIONS, FLAVOR, KEY_ADDED, OBSM_KEY

logger = logging.getLogger(__name__)


class ClusteringTask(BaseTask):
    """Task for evaluating clustering performance against ground truth labels.

    This task performs clustering on embeddings and evaluates the results
    using multiple clustering metrics (ARI and NMI).

    Args:
        label_key (str): Key to access ground truth labels in metadata
        random_seed (int): Random seed for reproducibility
    """

    def __init__(
        self,
        label_key: str,
        n_iterations: int = N_ITERATIONS,
        flavor: str = FLAVOR,
        key_added: str = KEY_ADDED,
        *,
        random_seed: int = RANDOM_SEED,
    ):
        super().__init__(random_seed=random_seed)
        self.display_name = "clustering"
        self.label_key = label_key
        self.n_iterations = n_iterations
        self.flavor = flavor
        self.key_added = key_added

    def _run_task(self, data: BaseDataset, **kwargs):
        """Runs clustering on the embedding data.

        Performs clustering and stores results for metric computation.

        Args:
            data: Dataset containing embedding and ground truth labels
        """
        # FIXME BYODATASET: decouple AnnData
        # Get anndata object and embedding
        adata = data.adata

        # Store labels and generate clusters
        input_labels = adata.obs[self.label_key].values

        predicted_labels = cluster_embedding(
            adata,
            obsm_key=OBSM_KEY,
            random_seed=self.random_seed,
            n_iterations=self.n_iterations,
            flavor=self.flavor,
            key_added=self.key_added,
        )

        return {
            "input_labels": input_labels,
            "predicted_labels": predicted_labels,
        }

    def _compute_metrics(
        self, input_labels: np.ndarray, predicted_labels: np.ndarray, **kwargs
    ) -> List[MetricResult]:
        """Computes clustering evaluation metrics.

        Returns:
            List of MetricResult objects containing ARI and NMI scores
        """
        return [
            MetricResult(
                metric_type=metric_type,
                value=metrics_registry.compute(
                    metric_type,
                    labels_true=input_labels,
                    labels_pred=predicted_labels,
                ),
                params={},
            )
            for metric_type in [
                MetricType.ADJUSTED_RAND_INDEX,
                MetricType.NORMALIZED_MUTUAL_INFO,
            ]
        ]
