from typing import Dict, Set, List
import numpy as np
from scib_metrics import silhouette_batch

from ..base import BaseTask
from ..utils import compute_entropy_per_cell
from ...datasets.single_cell import SingleCellDataset
from ...datasets.types import DataType


class CrossSpeciesIntegrationTask(BaseTask):
    def __init__(self, label_key: str):
        self.label_key = label_key

    @property
    def required_inputs(self) -> Set[DataType]:
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}

    @property
    def requires_multiple_datasets(self) -> bool:
        return True

    def _run_task(self, data: List[SingleCellDataset]):
        self.embedding = np.vstack([d.get_output(DataType.EMBEDDING) for d in data])
        self.labels = np.concatenate(
            [d.get_input(DataType.METADATA)[self.label_key] for d in data]
        )
        self.species = np.concatenate(
            [[d.organism.name] * d.adata.shape[0] for d in data]
        )

        return data

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "entropy_per_cell": compute_entropy_per_cell(self.embedding, self.species),
            "silhouette_score": silhouette_batch(
                self.embedding, self.labels, self.species
            ),
        }
