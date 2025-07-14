import logging
from typing import List
import pandas as pd
import anndata as ad

from czbenchmarks.types import ListLike

from .types import CellRepresentation
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from .base import Task
from .utils import cluster_embedding
from .constants import N_ITERATIONS, FLAVOR, KEY_ADDED
from ..constants import RANDOM_SEED

logger = logging.getLogger(__name__)


class ClusteringTask(Task):
    """Task for evaluating clustering performance against ground truth labels.

    This task performs clustering on embeddings and evaluates the results
    using multiple clustering metrics (ARI and NMI).

    Args:
        label_key (str): Key to access ground truth labels in metadata
        random_seed (int): Random seed for reproducibility
    """

    def __init__(
        self,
        *,
        random_seed: int = RANDOM_SEED,
    ):
        super().__init__(random_seed=random_seed)
        self.display_name = "clustering"

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        obs: pd.DataFrame,
        use_rep: str = "X",
        n_iterations: int = N_ITERATIONS,
        flavor: str = FLAVOR,
        key_added: str = KEY_ADDED,
        **kwargs,
    ) -> dict:
        """Runs clustering on the cell representation.

        Performs clustering and stores results for metric computation.

        Args:
            cell_representation: gene expression data or embedding for task
            obs: Obs dataframe
            use_rep: Use representation, default is "X"
            n_iterations: Number of iterations, default is N_ITERATIONS
            flavor: Flavor, default is FLAVOR
            key_added: Key added, default is KEY_ADDED
        """

        # Create the AnnData object
        adata = ad.AnnData(X=cell_representation, obs=obs)

        predicted_labels = cluster_embedding(
            adata,
            use_rep=use_rep,
            random_seed=self.random_seed,
            n_iterations=n_iterations,
            flavor=flavor,
            key_added=key_added,
        )

        return {
            "predicted_labels": predicted_labels,
        }

    def _compute_metrics(
        self, input_labels: ListLike, predicted_labels: ListLike, **kwargs
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
