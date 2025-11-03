import logging
from typing import Annotated, List, Literal
from pydantic import Field, field_validator

import anndata as ad
import pandas as pd

from czbenchmarks.types import ListLike

from ..constants import RANDOM_SEED
from ..metrics.types import MetricResult, MetricType
from .constants import FLAVOR, KEY_ADDED, N_ITERATIONS
from .task import PCABaselineInput, Task, TaskInput, TaskOutput
from .types import CellRepresentation
from .utils import cluster_embedding

logger = logging.getLogger(__name__)


class ClusteringTaskInput(TaskInput):
    obs: Annotated[
        pd.DataFrame,
        Field(
            description="Cell metadata DataFrame (e.g. the `obs` from an AnnData object)."
        ),
    ]
    input_labels: Annotated[
        ListLike,
        Field(
            description="Ground truth labels for metric calculation (e.g. `obs.cell_type` from an AnnData object)."
        ),
    ]
    use_rep: Annotated[
        str,
        Field(
            description="Data representation to use for clustering (e.g. the 'X' or obsm['X_pca'] from an AnnData object)."
        ),
    ] = "X"
    n_iterations: Annotated[
        int, Field(description="Number of iterations for the Leiden algorithm.")
    ] = N_ITERATIONS
    flavor: Annotated[
        Literal["leidenalg", "igraph"],
        Field(description="Algorithm for Leiden community detection."),
    ] = FLAVOR
    key_added: Annotated[
        str,
        Field(description="Key in AnnData.obs where cluster assignments are stored."),
    ] = KEY_ADDED

    @field_validator("n_iterations")
    @classmethod
    def _must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_iterations must be a positive integer.")
        return v


class ClusteringOutput(TaskOutput):
    """Output for clustering task."""

    predicted_labels: List[int]  # Predicted cluster labels


class ClusteringTask(Task):
    """Task for evaluating clustering performance against ground truth labels.

    This task performs clustering on embeddings and evaluates the results
    using multiple clustering metrics (ARI and NMI).
    """

    display_name = "Clustering"
    description = "Evaluate clustering performance against ground truth labels using ARI and NMI metrics."
    input_model = ClusteringTaskInput
    baseline_model = PCABaselineInput

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
    ) -> ClusteringOutput:
        """Runs clustering on the cell representation.

        Performs clustering and stores results for metric computation.

        Args:
            cell_representation: gene expression data or embedding for task
            task_input: Pydantic model with inputs for the task
        Returns:
            ClusteringOutput: Pydantic model with predicted cluster labels
        """
        logger.debug(
            f"ClusteringTask._run_task: cell_representation shape={cell_representation.shape}"
        )
        logger.debug(
            f"ClusteringTask._run_task: use_rep={task_input.use_rep}, flavor={task_input.flavor}"
        )

        # Create the AnnData object
        adata = ad.AnnData(
            X=cell_representation,
            obs=task_input.obs,
        )
        logger.debug(f"ClusteringTask: Created AnnData with shape {adata.shape}")

        predicted_labels = cluster_embedding(
            adata,
            use_rep=task_input.use_rep,
            random_seed=self.random_seed,
            n_iterations=task_input.n_iterations,
            flavor=task_input.flavor,
            key_added=task_input.key_added,
        )
        logger.debug(
            f"ClusteringTask: Generated {len(predicted_labels)} cluster labels"
        )

        return ClusteringOutput(predicted_labels=predicted_labels)

    def _compute_metrics(
        self,
        task_input: ClusteringTaskInput,
        task_output: ClusteringOutput,
    ) -> List[MetricResult]:
        """Computes clustering evaluation metrics.

        Args:
            task_input: Pydantic model with inputs for the task
            task_output: Pydantic model with outputs from _run_task

        Returns:
            List of MetricResult objects containing ARI and NMI scores
        """
        logger.debug("ClusteringTask._compute_metrics: Computing ARI and NMI metrics")

        from ..metrics import metrics_registry

        predicted_labels = task_output.predicted_labels
        results = [
            MetricResult(
                metric_type=metric_type,
                value=metrics_registry.compute(
                    metric_type,
                    labels_true=task_input.input_labels,
                    labels_pred=predicted_labels,
                ),
                params={},
            )
            for metric_type in [
                MetricType.ADJUSTED_RAND_INDEX,
                MetricType.NORMALIZED_MUTUAL_INFO,
            ]
        ]
        logger.debug(
            f"ClusteringTask._compute_metrics: Computed {len(results)} metrics"
        )
        return results
