from typing import List

import numpy as np

from ...constants import RANDOM_SEED
from ..base import BaseTask
from ...datasets.types import CellRepresentation, ListLike, Organism
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ..types import TaskInput, MetricInput, TaskOutput


__all__ = [
    "CrossSpeciesIntegrationTaskInput",
    "CrossSpeciesIntegrationMetricInput",
    "CrossSpeciesIntegrationOutput",
    "CrossSpeciesIntegrationTask",
]


class CrossSpeciesIntegrationTaskInput(TaskInput):
    """Pydantic model for CrossSpeciesIntegrationTask inputs."""

    labels: List[ListLike]
    organism_list: List[Organism]


class CrossSpeciesIntegrationMetricInput(MetricInput):
    """Pydantic model for CrossSpeciesIntegrationTask metric inputs."""

    pass


class CrossSpeciesIntegrationOutput(TaskOutput):
    """Output for cross-species integration task."""
    cell_representation: CellRepresentation
    labels: ListLike
    species: ListLike


class CrossSpeciesIntegrationTask(BaseTask):
    """Task for evaluating cross-species integration quality.

    This task computes metrics to assess how well different species' data are integrated
    in the embedding space while preserving biological signals. It operates on multiple
    datasets from different species.

    Args:
        random_seed (int): Random seed for reproducibility
    """

    display_name = "cross-species integration"

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.requires_multiple_datasets = True

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: CrossSpeciesIntegrationTaskInput,
    ) -> CrossSpeciesIntegrationOutput:
        """Runs the cross-species integration evaluation task.

        Gets embedding coordinates and labels from multiple datasets and combines them
        for metric computation.

        Args:
            cell_representation: list of cell representations for the task
            task_input: Pydantic model with inputs for the task

        Returns:
            CrossSpeciesIntegrationOutput: Pydantic model with combined data and labels
        """
        # FIXME BYODATASETdatasets should be concatenated to align along genes?
        # This operation is safe because requires_multiple_datasets is True
        cell_representation = np.vstack(cell_representation)

        # FIXME BYODATASET move this into validation
        if len(set(task_input.organism_list)) < 2:
            raise AssertionError(
                "At least two organisms are required for cross-species integration "
                f"but got {len(set(task_input.organism_list))} : {{set(task_input.organism_list)}}"
            )

        species = np.concatenate(
            [
                [
                    str(organism),
                ]
                * len(label)
                for organism, label in zip(
                    task_input.organism_list, task_input.labels
                )
            ]
        )
        labels = np.concatenate(task_input.labels)

        if (len(cell_representation) != len(species)) or (
            len(species) != len(labels)
        ):
            raise AssertionError(
                "Cell representation, species, and labels must have the same shape"
            )

        return CrossSpeciesIntegrationOutput(
            cell_representation=cell_representation,
            labels=labels,
            species=species,
        )

    def _compute_metrics(
        self,
        task_output: CrossSpeciesIntegrationOutput,
        metric_input: CrossSpeciesIntegrationMetricInput,
    ) -> List[MetricResult]:
        """Computes batch integration quality metrics.

        Args:
            task_output: Pydantic model with outputs from _run_task
            metric_input: Pydantic model with inputs for the metrics

        Returns:
            List of MetricResult objects containing entropy per cell and
            batch-aware silhouette scores
        """

        entropy_per_cell_metric = MetricType.ENTROPY_PER_CELL
        silhouette_batch_metric = MetricType.BATCH_SILHOUETTE
        cell_representation = task_output.cell_representation
        labels = task_output.labels
        species = task_output.species

        return [
            MetricResult(
                metric_type=entropy_per_cell_metric,
                value=metrics_registry.compute(
                    entropy_per_cell_metric,
                    X=cell_representation,
                    labels=species,
                    random_seed=self.random_seed,
                ),
            ),
            MetricResult(
                metric_type=silhouette_batch_metric,
                value=metrics_registry.compute(
                    silhouette_batch_metric,
                    X=cell_representation,
                    labels=labels,
                    batch=species,
                ),
            ),
        ]

    def compute_baseline(self, **kwargs):
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
