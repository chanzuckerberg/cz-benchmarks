from typing import List

import numpy as np

from ..constants import RANDOM_SEED
from ...datasets import SingleCellDataset, DataType
from ..base import BaseTask
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ...models.types import ModelType


class CrossSpeciesIntegrationTask(BaseTask):
    """Task for evaluating cross-species integration quality.

    This task computes metrics to assess how well different species' data are integrated
    in the embedding space while preserving biological signals. It operates on multiple
    datasets from different species.

    Args:
        label_key: Key to access ground truth cell type labels in metadata
        random_seed (int): Random seed for reproducibility
    """

    def __init__(self, label_key: str, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.display_name = "cross-species integration"
        self.requires_multiple_datasets = True
        self.label_key = label_key

    def _run_task(self, data: List[SingleCellDataset], **kwargs) -> dict:
        """Runs the cross-species integration evaluation task.

        Gets embedding coordinates and labels from multiple datasets and combines them
        for metric computation.

        Args:
            data: List of datasets containing embeddings and labels from different
                  species

        Returns:
            Dictionary of labels and species
        """
        labels = np.concatenate([d.adata.obs[self.label_key].values for d in data])
        species = np.concatenate([[d.organism.name] * d.adata.shape[0] for d in data])

        return {
            "labels": labels,
            "species": species,
        }

    def _compute_metrics(
        self, embedding: np.ndarray, labels: np.ndarray, species: np.ndarray
    ) -> List[MetricResult]:
        """Computes batch integration quality metrics.

        Args:
            embedding: embedding to use for the task
            labels: labels to use for the task
            species: species to use for the task

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
                    X=embedding,
                    labels=species,
                    random_seed=self.random_seed,
                ),
            ),
            MetricResult(
                metric_type=silhouette_batch_metric,
                value=metrics_registry.compute(
                    silhouette_batch_metric,
                    X=embedding,
                    labels=labels,
                    batch=species,
                ),
            ),
        ]

    def set_baseline(self, data: List[SingleCellDataset], **kwargs):
        """Set a baseline embedding for cross-species integration.

        This method is not implemented for cross-species integration tasks
        as standard preprocessing workflows are not directly applicable
        across different species.

        Args:
            data: List of SingleCellDataset objects from different species
            **kwargs: Additional arguments passed to run_standard_scrna_workflow

        Raises:
            NotImplementedError: Always raised as baseline is not implemented
        """
        raise NotImplementedError(
            "Baseline not implemented for cross-species integration"
        )
