from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from .dataset import Dataset
from .types import Organism
import logging

logger = logging.getLogger(__name__)


class SingleCellDataset(Dataset):
    """Abstract base class for single cell datasets containing gene expression data."""
    
    adata: ad.AnnData
    
    def __init__(
        self,
        name: str,
        path: Path,
        organism: Organism,
    ):
        super().__init__(name, path, organism)

    def load_data(self) -> None:
        """Load the dataset from the path."""
        # FIXME: Update as needed when cache PR is merged
        self.adata = ad.read_h5ad(self.path)

    @property
    def adata(self) -> ad.AnnData:
        """Getter method for the AnnData object."""
        if not hasattr(self, '_adata'):
            raise AttributeError("The AnnData object has not been loaded yet. Call 'load_data()' first.")
        return self._adata

    @adata.setter
    def adata(self, value: ad.AnnData) -> None:
        """Setter method for the AnnData object."""
        self._adata = value

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


