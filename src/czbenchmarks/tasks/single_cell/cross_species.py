from typing import List

import numpy as np

from ...constants import RANDOM_SEED
from ..base import BaseTask
from ...datasets.types import Embedding, ListLike
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType


class CrossSpeciesIntegrationTask(BaseTask):
    """Task for evaluating cross-species integration quality.

    This task computes metrics to assess how well different species' data are integrated
    in the embedding space while preserving biological signals. It operates on multiple
    datasets from different species.

    Args:
        random_seed (int): Random seed for reproducibility
    """

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.display_name = "cross-species integration"
        self.requires_multiple_datasets = True

    def _run_task(self, labels: List[ListLike], organism_list: List[str], **kwargs) -> dict:
        """Runs the cross-species integration evaluation task.

        Gets embedding coordinates and labels from multiple datasets and combines them
        for metric computation.

        Args:
            labels: labels to use for the task
            organism_list: list of organisms to use for the task

        Returns:
            Dictionary of labels and species
        """
        species = np.concatenate([organism] * len(label) for organism, label in zip(organism_list, labels))
        labels = np.concatenate(labels)
        
        assert species.shape == labels.shape, AssertionError("Species and labels must have the same shape")
        
        return {
            "labels": labels,
            "species": species,
        }

    def _compute_metrics(
        self, embedding: Embedding, labels: ListLike, species: ListLike
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

    def set_baseline(self, **kwargs):
        """Set a baseline embedding for cross-species integration.

        This method is not implemented for cross-species integration tasks
        as standard preprocessing workflows are not directly applicable
        across different species.

        Raises:
            NotImplementedError: Always raised as baseline is not implemented
        """
        raise NotImplementedError(
            "Baseline not implemented for cross-species integration"
        )
