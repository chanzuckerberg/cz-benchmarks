from typing import Dict, List, Set

import numpy as np

from ...datasets.single_cell import SingleCellDataset
from ...datasets.types import DataType
from ..base import BaseTask
from ...metrics import MetricType, metrics


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

    def _run_task(self, data: List[SingleCellDataset]):
        """Runs the cross-species integration evaluation task.

        Gets embedding coordinates and labels from multiple datasets and combines them
        for metric computation.

        Args:
            data: List of datasets containing embeddings and labels from different
                  species
        """
        self.embedding = np.vstack([d.get_output(DataType.EMBEDDING) for d in data])
        self.labels = np.concatenate(
            [d.get_input(DataType.METADATA)[self.label_key] for d in data]
        )
        self.species = np.concatenate(
            [[d.organism.name] * d.adata.shape[0] for d in data]
        )

        return data

    def _compute_metrics(self) -> Dict[str, float]:
        """Computes cross-species integration quality metrics.

        Returns:
            Dictionary containing entropy per cell and species-aware silhouette scores
        """

        entropy_per_cell_metric = MetricType.ENTROPY_PER_CELL
        silhouette_batch_metric = MetricType.BATCH_SILHOUETTE

        return {
            entropy_per_cell_metric.value: metrics.compute(
                entropy_per_cell_metric,
                X=self.embedding,
                labels=self.species,
            ),
            silhouette_batch_metric.value: metrics.compute(
                silhouette_batch_metric,
                X=self.embedding,
                labels=self.labels,
                batch_labels=self.species,
            ),
        }
