import logging
from typing import List, Literal
import pandas as pd
import anndata as ad

from ..datasets.types import CellRepresentation, ListLike
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from .base import BaseTask
from .utils import cluster_embedding
from .constants import N_ITERATIONS, FLAVOR, KEY_ADDED
from ..constants import RANDOM_SEED
from .types import TaskInput, MetricInput


logger = logging.getLogger(__name__)


__all__ = [
    "ClusteringTaskInput",
    "ClusteringMetricInput",
    "ClusteringTask",
]


class ClusteringTaskInput(TaskInput):
    obs: pd.DataFrame
    use_rep: str = "X"
    n_iterations: int = N_ITERATIONS
    flavor: Literal["leidenalg", "igraph"] = FLAVOR
    key_added: str = KEY_ADDED


class ClusteringMetricInput(MetricInput):
    input_labels: ListLike


class ClusteringTask(BaseTask):
    """Task for evaluating clustering performance against ground truth labels.

    This task performs clustering on embeddings and evaluates the results
    using multiple clustering metrics (ARI and NMI).

    Args:
        random_seed (int): Random seed for reproducibility
    """

    display_name = "clustering"

    def __init__(
        self,
        *,
        random_seed: int = RANDOM_SEED,
    ):
        super().__init__(random_seed=random_seed)

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: ClusteringTaskInput,
    ) -> dict:
        """Runs clustering on the cell representation.

        Performs clustering and stores results for metric computation.

        Args:
            cell_representation: gene expression data or embedding for task
            task_input: Pydantic model with inputs for the task
        """

        # Create the AnnData object
        adata = ad.AnnData(
            X=cell_representation,
            obs=task_input.obs,
        )

        predicted_labels = cluster_embedding(
            adata,
            use_rep=task_input.use_rep,
            random_seed=self.random_seed,
            n_iterations=task_input.n_iterations,
            flavor=task_input.flavor,
            key_added=task_input.key_added,
        )

        return {
            "predicted_labels": predicted_labels,
        }

    def _compute_metrics(
        self, task_output: dict, metric_input: ClusteringMetricInput
    ) -> List[MetricResult]:
        """Computes clustering evaluation metrics.

        Returns:
            List of MetricResult objects containing ARI and NMI scores
        """
        predicted_labels = task_output["predicted_labels"]
        return [
            MetricResult(
                metric_type=metric_type,
                value=metrics_registry.compute(
                    metric_type,
                    labels_true=metric_input.input_labels,
                    labels_pred=predicted_labels,
                ),
                params={},
            )
            for metric_type in [
                MetricType.ADJUSTED_RAND_INDEX,
                MetricType.NORMALIZED_MUTUAL_INFO,
            ]
        ]
