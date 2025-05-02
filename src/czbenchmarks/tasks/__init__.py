from .clustering import ClusteringTask
from .embedding import EmbeddingTask
from .label_prediction import MetadataLabelPredictionTask
from .integration import BatchIntegrationTask
from .single_cell import PerturbationTask, CrossSpeciesIntegrationTask

__all__ = [
    "EmbeddingTask",
    "ClusteringTask",
    "MetadataLabelPredictionTask",
    "BatchIntegrationTask",
    "PerturbationTask",
    "CrossSpeciesIntegrationTask",
]

def _enable_gpu_acceleration():
    try:
        import cuml.accel
        cuml.accel.install()
        return True
    except ImportError:
        return False

HAS_GPU_ACCEL = _enable_gpu_acceleration()
