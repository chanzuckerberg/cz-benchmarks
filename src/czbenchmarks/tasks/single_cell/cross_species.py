from typing import List, Set

import numpy as np

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
            required output types from models this task to run (embedding coordinates)
        """
        return {DataType.EMBEDDING}

    @property
    def requires_multiple_datasets(self) -> bool:
        """Whether this task requires multiple datasets.

        Returns:
            True as this task compares data across species
        """
        return True

    def _run_task(self, data: List[SingleCellDataset], model_type: ModelType):
        """Runs the cross-species integration evaluation task.

        Gets embedding coordinates and labels from multiple datasets and combines them
        for metric computation.

        Args:
            data: List of datasets containing embeddings and labels from different
                  species
        """
        self.embedding = np.vstack(
            [d.get_output(model_type, DataType.EMBEDDING) for d in data]
        )
        self.labels = np.concatenate(
            [d.get_input(DataType.METADATA)[self.label_key] for d in data]
        )
        self.species = np.concatenate(
            [[d.organism.name] * d.adata.shape[0] for d in data]
        )

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
                    labels=self.species,
                ),
            ),
            MetricResult(
                metric_type=silhouette_batch_metric,
                value=metrics_registry.compute(
                    silhouette_batch_metric,
                    X=self.embedding,
                    labels=self.labels,
                    batch=self.species,
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
