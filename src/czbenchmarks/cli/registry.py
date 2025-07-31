from typing import Dict

from .types import TaskDefinition
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

# The registry is now clean and minimal. It only declares WHAT a task is,
# not HOW its parameters are configured.
TASK_REGISTRY: Dict[str, TaskDefinition] = {
    "clustering": TaskDefinition(
        task_class=ClusteringTask,
        input_model=ClusteringTaskInput,
        display_name="Clustering",
        description="Evaluate clustering performance against ground truth labels.",
    ),
    "embedding": TaskDefinition(
        task_class=EmbeddingTask,
        input_model=EmbeddingTaskInput,
        display_name="Embedding Quality",
        description="Evaluate embedding quality using silhouette score.",
    ),
    "integration": TaskDefinition(
        task_class=BatchIntegrationTask,
        input_model=BatchIntegrationTaskInput,
        display_name="Batch Integration",
        description="Evaluate batch integration quality.",
    ),
    "label_prediction": TaskDefinition(
        task_class=MetadataLabelPredictionTask,
        input_model=MetadataLabelPredictionTaskInput,
        display_name="Label Prediction",
        description="Predict cell labels from embeddings using cross-validation.",
    ),
    "perturbation": TaskDefinition(
        task_class=PerturbationTask,
        input_model=PerturbationTaskInput,
        display_name="Perturbation Prediction",
        description="Evaluate perturbation prediction against ground truth.",
    ),
    "cross_species_integration": TaskDefinition(
        task_class=CrossSpeciesIntegrationTask,
        input_model=CrossSpeciesIntegrationTaskInput,
        display_name="Cross-Species Integration",
        description="Evaluate integration of data from multiple species.",
        requires_multiple_datasets=True,
    ),
}


def get_task_def(name: str) -> TaskDefinition:
    """Retrieve a task definition from the registry."""
    task_def = TASK_REGISTRY.get(name)
    if not task_def:
        raise ValueError(f"Task '{name}' not found in registry.")
    return task_def
