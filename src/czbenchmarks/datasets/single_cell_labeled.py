from pathlib import Path
import anndata as ad
import pandas as pd
import numpy as np

from czbenchmarks.datasets.single_cell import SingleCellDataset
from .dataset import Dataset
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
        # FIXME: Update as needed when cache PR is merged
        super().load_data()
        self.labels = self.adata.obs["cell_type"]

    def store_task_inputs(self) -> Path:
        """Store task-specific inputs, such as cell type annotations."""

        buffer = io.StringIO()
        self.labels.to_json(buffer)

        return self._store_task_input("cell_types.json", buffer.getvalue())


    # FIXME VALIDATION: move to validation class?
    def _validate(self) -> None:
        if not isinstance(self.organism, Organism):
            raise ValueError("Organism is not a valid Organism enum")

        var = all(self.adata.var_names.str.startswith(self.organism.prefix))

        # Check if data contains non-integer or negative values
        data = (
            self.adata.X.data
            if hasattr(self.adata.X, "data")
            and not isinstance(self.adata.X, np.ndarray)
            else self.adata.X
        )
        if np.any(np.mod(data, 1) != 0) or np.any(data < 0):
            logger.warning(
                "Dataset X matrix does not contain raw counts."
                " Some models may require raw counts as input."
                " Check the corresponding model card for more details."
            )

        if not var:
            if "ensembl_id" in self.adata.var.columns:
                self.adata.var_names = pd.Index(list(self.adata.var["ensembl_id"]))
                var = all(self.adata.var_names.str.startswith(self.organism.prefix))

        if not var:
            raise ValueError(
                "Dataset does not contain valid gene names. Gene names must"
                f" start with {self.organism.prefix} and be stored in either"
                f" adata.var_names or adata.var['ensembl_id']."
            )


