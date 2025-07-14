from .task import Task
from .clustering import ClusteringTask
from .embedding import EmbeddingTask
from .label_prediction import MetadataLabelPredictionTask
from .integration import BatchIntegrationTask
from .single_cell import PerturbationTask, CrossSpeciesIntegrationTask

__all__ = [
    "Task",
    "EmbeddingTask",
    "ClusteringTask",
    "MetadataLabelPredictionTask",
    "BatchIntegrationTask",
    "PerturbationTask",
    "CrossSpeciesIntegrationTask",
]
