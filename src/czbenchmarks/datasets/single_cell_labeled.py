from pathlib import Path
from typing import Optional
import pandas as pd

from czbenchmarks.datasets.single_cell import SingleCellDataset
from .types import Organism
import logging
import io

logger = logging.getLogger(__name__)


class SingleCellLabeledDataset(SingleCellDataset):
    """Single cell dataset containing gene expression data and a label column containing the expected prediction values for each cell."""
    
    labels: pd.Series
    label_column_key: str

    def __init__(
        self,
        path: Path,
        organism: Organism,
        label_column_key: str = "cell_type",
        task_inputs_dir: Optional[Path] = None,
    ):
        super().__init__("single_cell_labeled", path, organism, task_inputs_dir)
        self.label_column_key = label_column_key

    def load_data(self) -> None:
        """Load the dataset from the path."""
        super().load_data()
        self.labels = self.adata.obs[self.label_column_key]

    def store_task_inputs(self) -> Path:
        """Store task-specific inputs, such as cell type annotations."""

        buffer = io.StringIO()
        self.labels.to_json(buffer)

        filename = f"labels_{self.label_column_key}.json"
        self._store_task_input(filename, buffer.getvalue())
        return self.task_inputs_dir


    # FIXME VALIDATION: move to validation class?
    def _validate(self) -> None:
        super()._validate()

        # TODO: This check needs to occur after loading the data, but before task inputs are extracted.
        if self.label_column_key not in self.adata.obs.columns:
            raise ValueError(f"Dataset does not contain '{self.label_column_key}' column in obs.")
