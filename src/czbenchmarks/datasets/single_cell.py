from pathlib import Path
from typing import Optional
import anndata as ad
import numpy as np
import pandas as pd
from .dataset import Dataset
from .types import Organism
import logging

logger = logging.getLogger(__name__)


class SingleCellDataset(Dataset):
    """
    Abstract base class for single cell datasets containing gene expression data.

    Handles loading and validation of AnnData objects with the following requirements:
    - Must have gene names in `adata.var['ensembl_id']`
    - Gene names must start with the organism prefix (e.g., "ENSG" for human)
    - Must contain raw counts in `adata.X` (non-negative integers)
    - Should be stored in H5AD format
    """

    adata: ad.AnnData

    def __init__(
        self,
        dataset_type_name: str,
        path: Path,
        organism: Organism,
        task_inputs_dir: Optional[Path] = None,
    ):
        super().__init__(dataset_type_name, path, organism, task_inputs_dir)

    def load_data(self) -> None:
        """Load the dataset from the path."""
        # FIXME: Update as needed when cache PR is merged
        self.adata = ad.read_h5ad(self.path)

    def _validate(self) -> None:
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
