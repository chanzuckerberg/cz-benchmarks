from pathlib import Path
import pandas as pd

from czbenchmarks.datasets.single_cell import SingleCellDataset
from .types import Organism
import logging
import io

logger = logging.getLogger(__name__)


class SingleCellLabeledDataset(SingleCellDataset):
    """Single cell dataset containing gene expression data and "cell_type" obs label column for cells."""
    
    labels: pd.Series

    def __init__(
        self,
        path: Path,
        organism: Organism,
    ):
        super().__init__("single_cell_labeled", path, organism)

    def load_data(self) -> None:
        """Load the dataset from the path."""
        super().load_data()
        self.labels = self.adata.obs["cell_type"]

    def store_task_inputs(self) -> Path:
        """Store task-specific inputs, such as cell type annotations."""

        buffer = io.StringIO()
        self.labels.to_json(buffer)

        return self._store_task_input("cell_types.json", buffer.getvalue())


    # FIXME VALIDATION: move to validation class?
    def _validate(self) -> None:
        super()._validate()

        # TODO: This check needs to occur after loading the data, but before task inputs are extracted.
        if "cell_type" not in self.adata.obs.columns:
            raise ValueError("Dataset does not contain 'cell_type' column in obs.")
