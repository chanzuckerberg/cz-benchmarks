from .clustering import ClusteringTask
from .embedding import EmbeddingTask
from .integration import BatchIntegrationTask
from .label_prediction import MetadataLabelPredictionTask
from .single_cell.perturbation import (
    PerturbationTask as SingleCellPerturbationTask,
)

__all__ = [
    "EmbeddingTask",
    "ClusteringTask",
    "MetadataLabelPredictionTask",
    "SingleCellPerturbationTask",
    "BatchIntegrationTask",
]
