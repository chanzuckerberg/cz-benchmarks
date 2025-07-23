from typing import Dict

from .types import TaskDefinition, TaskParameter
from ..tasks import (
    ClusteringTask,
    ClusteringTaskInput,
    EmbeddingTask,
    EmbeddingTaskInput,
    BatchIntegrationTask,
    BatchIntegrationTaskInput,
    MetadataLabelPredictionTask,
    MetadataLabelPredictionTaskInput,
)
from ..tasks.single_cell import (
    CrossSpeciesIntegrationTask,
    CrossSpeciesIntegrationTaskInput,
    PerturbationTask,
    PerturbationTaskInput,
)


TASK_REGISTRY: Dict[str, TaskDefinition] = {
    "clustering": TaskDefinition(
        task_class=ClusteringTask,
        input_model=ClusteringTaskInput,
        display_name="Clustering",
        description="Evaluate clustering performance against ground truth labels.",
        parameters=[], # Inputs like 'obs' and 'input_labels' are sourced directly from the dataset
    ),
    "embedding": TaskDefinition(
        task_class=EmbeddingTask,
        input_model=EmbeddingTaskInput,
        display_name="Embedding Quality",
        description="Evaluate embedding quality using silhouette score.",
        parameters=[], # 'input_labels' sourced from dataset
    ),
    "integration": TaskDefinition(
        task_class=BatchIntegrationTask,
        input_model=BatchIntegrationTaskInput,
        display_name="Batch Integration",
        description="Evaluate batch integration quality.",
        parameters=[
            TaskParameter(name="batch_key", type=str, help="Key in adata.obs for batch labels.", required=True),
        ],
    ),
    "label_prediction": TaskDefinition(
        task_class=MetadataLabelPredictionTask,
        input_model=MetadataLabelPredictionTaskInput,
        display_name="Label Prediction",
        description="Predict cell labels from embeddings using cross-validation.",
        parameters=[
            TaskParameter(name="n_folds", type=int, help="Number of cross-validation folds.", default=5),
            TaskParameter(name="min_class_size", type=int, help="Minimum samples per class for filtering.", default=10),
        ],
    ),
    "perturbation": TaskDefinition(
        task_class=PerturbationTask,
        input_model=PerturbationTaskInput,
        display_name="Perturbation Prediction",
        description="Evaluate perturbation prediction against ground truth.",
        parameters=[
            TaskParameter(name="gene_pert", type=str, help="Gene perturbation to evaluate.", required=True),
        ],
        baseline_parameters=[
            TaskParameter(name="type", type=str, default="median", help="Type of baseline to compute ('mean' or 'median').")
        ]
    ),
    "cross_species_integration": TaskDefinition(
        task_class=CrossSpeciesIntegrationTask,
        input_model=CrossSpeciesIntegrationTaskInput,
        display_name="Cross-Species Integration",
        description="Evaluate integration of data from multiple species.",
        parameters=[],
        requires_multiple_datasets=True,
    ),
}

def get_task_def(name: str) -> TaskDefinition:
    """Retrieve a task definition from the registry."""
    task_def = TASK_REGISTRY.get(name)
    if not task_def:
        raise ValueError(f"Task '{name}' not found in registry.")
    return task_def