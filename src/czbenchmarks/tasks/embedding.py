import logging
from typing import Annotated, List
from pydantic import Field

import scipy.sparse as sp

from ..constants import RANDOM_SEED
from ..metrics import metrics_registry
from ..metrics.types import MetricResult, MetricType
from ..tasks.types import CellRepresentation
from ..types import ListLike
from .task import PCABaselineInput, Task, TaskInput, TaskOutput

logger = logging.getLogger(__name__)


class EmbeddingTaskInput(TaskInput):
    """Pydantic model for EmbeddingTask inputs."""

    input_labels: Annotated[
        ListLike,
        Field(
            description="Ground truth labels for metric calculation (e.g., 'cell_type' or '@obs:cell_type')."
        ),
    ]


class EmbeddingOutput(TaskOutput):
    """Output for embedding task."""

    cell_representation: CellRepresentation  # The cell representation matrix


class EmbeddingTask(Task):
    """Task for evaluating cell representation quality using labeled data.

    This task computes quality metrics for cell representations using ground truth labels.
    Currently supports silhouette score evaluation.
    """

    display_name = "Embedding"
    description = "Evaluate cell representation quality using silhouette score with ground truth labels."
    input_model = EmbeddingTaskInput
    baseline_model = PCABaselineInput

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
        logger.debug(
            f"EmbeddingTask._run_task: cell_representation shape={cell_representation.shape}"
        )
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
        logger.debug("EmbeddingTask._compute_metrics: Computing silhouette score")
        metric_type = MetricType.SILHOUETTE_SCORE
        cell_representation = task_output.cell_representation

        # Convert sparse matrix to dense if needed for JAX compatibility in metrics
        if sp.issparse(cell_representation):
            logger.debug("EmbeddingTask: Converting sparse matrix to dense")
            cell_representation = cell_representation.toarray()

        logger.debug(
            f"EmbeddingTask: Computing metric on representation shape: {cell_representation.shape}"
        )
        result = [
            MetricResult(
                metric_type=metric_type,
                value=metrics_registry.compute(
                    metric_type,
                    X=cell_representation,
                    labels=task_input.input_labels,
                ),
            )
        ]
        logger.debug(
            f"EmbeddingTask._compute_metrics: Computed {len(result)} metric(s)"
        )
        return result
