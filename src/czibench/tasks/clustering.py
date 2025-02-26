import logging
from typing import Dict, Set

from ..datasets.sc import SingleCellDataset
from ..datasets.types import DataType
from ..metrics.clustering import adjusted_rand_index, normalized_mutual_info
from .base import BaseTask
from .utils import cluster_embedding

logger = logging.getLogger(__name__)


class ClusteringTask(BaseTask):
    def __init__(self, label_key: str):
        self.label_key = label_key

    @property
    def required_inputs(self) -> Set[DataType]:
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}

    def _run_task(self, data: SingleCellDataset) -> SingleCellDataset:
        adata = data.adata
        adata.obsm["emb"] = data.get_output(DataType.EMBEDDING)
        self.input_labels = data.get_input(DataType.METADATA)[self.label_key]
        self.predicted_labels = cluster_embedding(adata, obsm_key="emb")
        return data

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "adjusted_rand_index": adjusted_rand_index(
                self.input_labels, self.predicted_labels
            ),
            "normalized_mutual_info": normalized_mutual_info(
                self.input_labels, self.predicted_labels
            ),
        }
