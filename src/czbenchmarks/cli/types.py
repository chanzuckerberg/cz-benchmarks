from czbenchmarks.tasks.clustering import ClusteringTask, ClusteringTaskInput
from czbenchmarks.tasks.embedding import EmbeddingTask, EmbeddingTaskInput
from czbenchmarks.tasks.label_prediction import (
    MetadataLabelPredictionTask,
    MetadataLabelPredictionTaskInput,
)
from czbenchmarks.tasks.integration import (
    BatchIntegrationTask,
    BatchIntegrationTaskInput,
)
from czbenchmarks.tasks.single_cell.cross_species import (
    CrossSpeciesIntegrationTask,
    CrossSpeciesIntegrationTaskInput,
)
from czbenchmarks.tasks.single_cell.perturbation import (
    PerturbationTask,
    PerturbationTaskInput,
)


# Central Task Registry: Defines tasks, their classes, and required/optional parameters.
TASK_REGISTRY = {
    "clustering": {
        "task_class": ClusteringTask,
        "input_class": ClusteringTaskInput,
        "params": {"input_labels": "obs_key", "n_iterations": int, "flavor": str},
        "help": "Evaluate clustering (ARI, NMI). Requires --obs-key.",
    },
    "embedding": {
        "task_class": EmbeddingTask,
        "input_class": EmbeddingTaskInput,
        "params": {"input_labels": "obs_key"},
        "help": "Evaluate embedding quality (silhouette score). Requires --obs-key.",
    },
    "label-prediction": {
        "task_class": MetadataLabelPredictionTask,
        "input_class": MetadataLabelPredictionTaskInput,
        "params": {"labels": "obs_key", "n_folds": int, "min_class_size": int},
        "help": "Evaluate label prediction (cross-validation). Requires --obs-key.",
    },
    "integration": {
        "task_class": BatchIntegrationTask,
        "input_class": BatchIntegrationTaskInput,
        "params": {"labels": "obs_key", "batch_labels": "batch_key"},
        "help": "Evaluate batch integration. Requires --obs-key and --batch-key.",
    },
    "cross-species": {
        "task_class": CrossSpeciesIntegrationTask,
        "input_class": CrossSpeciesIntegrationTaskInput,
        "params": {"labels": "obs_key"},
        "requires_multiple_datasets": True,
        "help": "Evaluate cross-species integration. Requires multiple datasets and embeddings, plus --obs-key.",
    },
    "perturbation": {
        "task_class": PerturbationTask,
        "input_class": PerturbationTaskInput,
        "params": {"gene_pert": str},
        "help": "Evaluate perturbation prediction. Requires --task-param gene_pert=<gene_name>.",
    },
}
