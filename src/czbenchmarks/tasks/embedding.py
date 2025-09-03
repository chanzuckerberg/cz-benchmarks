import logging
from typing import List

from ..constants import RANDOM_SEED
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from ..tasks.types import CellRepresentation
from ..types import ListLike
from .task import Task, TaskInput, TaskOutput

logger = logging.getLogger(__name__)


class EmbeddingTaskInput(TaskInput):
    """Pydantic model for EmbeddingTask inputs."""

    input_labels: ListLike


class EmbeddingOutput(TaskOutput):
    """Output for embedding task."""

    cell_representation: CellRepresentation  # The cell representation matrix


class EmbeddingTask(Task):
    """Task for evaluating cell representation quality using labeled data.

    This task computes quality metrics for cell representations using ground truth labels.
    Currently supports silhouette score evaluation.

    Args:
        random_seed (int): Random seed for reproducibility
    """

    display_name = "embedding"
    description = "Evaluate cell representation quality using silhouette score with ground truth labels."
    input_model = EmbeddingTaskInput

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)

    def _run_task(
        self, cell_representation: CellRepresentation, _: EmbeddingTaskInput
    ) -> EmbeddingOutput:
        """Run the task's core computation.

        Args:
            cell_representation: gene expression data or embedding for task
            _: (unused) Pydantic model with inputs for the task
        Returns:
            EmbeddingOutput: Pydantic model with cell representation
        """
        return EmbeddingOutput(cell_representation=cell_representation)

    def _compute_metrics(
        self, task_input: EmbeddingTaskInput, task_output: EmbeddingOutput
    ) -> List[MetricResult]:
        """Computes cell representation quality metrics.

        Args:
            task_input: Pydantic model with inputs for the task
            task_output: Pydantic model with task outputs

        Returns:
            List of MetricResult objects containing silhouette score
        """
        metric_type = MetricType.SILHOUETTE_SCORE
        cell_representation = task_output.cell_representation
        return [
            MetricResult(
                metric_type=metric_type,
                value=metrics_registry.compute(
                    metric_type,
                    X=cell_representation,
                    labels=task_input.input_labels,
                ),
            )
        ]
