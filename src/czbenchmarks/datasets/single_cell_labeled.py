from pathlib import Path
from typing import Optional
import pandas as pd

from .single_cell import SingleCellDataset
from .types import Organism
import logging
import io

logger = logging.getLogger(__name__)


class SingleCellLabeledDataset(SingleCellDataset):
    """
    Single cell dataset containing gene expression data and a label column.

    This class extends `SingleCellDataset` to include a label column that contains
    the expected prediction values for each cell. The labels are extracted from
    the specified column in `adata.obs` and stored as a `pd.Series` in the `labels`
    attribute.

    Attributes:
        labels (pd.Series): Extracted labels for each cell.
        label_column_key (str): Key for the column in `adata.obs` containing the labels.
    """

    labels: pd.Series
    label_column_key: str

    def __init__(
        self,
        path: Path,
        organism: Organism,
        label_column_key: str = "cell_type",
        task_inputs_dir: Optional[Path] = None,
    ):
        """
        Initialize a SingleCellLabeledDataset instance.

        Args:
            path (Path): Path to the dataset file.
            organism (Organism): Enum value indicating the organism.
            label_column_key (str): Key for the column in `adata.obs` containing the labels.
                Defaults to "cell_type".
            task_inputs_dir (Optional[Path]): Directory for storing task-specific inputs.
        """
        super().__init__("single_cell_labeled", path, organism, task_inputs_dir)
        self.label_column_key = label_column_key

    def load_data(self) -> None:
        """
        Load the dataset and extract labels.

        This method loads the dataset using the parent class's `load_data` method
        and extracts the labels from the specified column in `adata.obs`.

        Populates:
            labels (pd.Series): Extracted labels for each cell.
        """
        super().load_data()
        self.labels = self.adata.obs[self.label_column_key]

    def store_task_inputs(self) -> Path:
        """
        Store task-specific inputs, such as cell type annotations.

        This method stores the extracted labels in a JSON file. The filename is
        dynamically generated based on the `label_column_key`.

        Returns:
            Path: Path to the directory storing the task input files.
        """
        buffer = io.StringIO()
        self.labels.to_json(buffer)

        filename = f"labels_{self.label_column_key}.json"
        self._store_task_input(filename, buffer.getvalue())
        return self.task_inputs_dir

    def _validate(self) -> None:
        """
        Perform dataset-specific validation.

        Validates that the specified `label_column_key` exists in `adata.obs.columns`.

        Raises:
            ValueError: If the `label_column_key` is not found in `adata.obs.columns`.
        """
        super()._validate()

        if self.label_column_key not in self.adata.obs.columns:
            raise ValueError(
                f"Dataset does not contain '{self.label_column_key}' column in obs."
            )
