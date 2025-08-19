from pathlib import Path
from typing import Optional, Literal
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
    - Must have gene names in `adata.var['ensembl_id']` or `adata.var_names`.
    - Gene names must start with the organism prefix (e.g., "ENSG" for human).
    - Must contain raw counts in `adata.X` (non-negative integers).
    - Should be stored in H5AD format.

    Attributes:
        adata (ad.AnnData): Loaded AnnData object containing gene expression data.
    """

    adata: ad.AnnData

    def __init__(
        self,
        dataset_type_name: str,
        path: Path,
        organism: Organism,
        task_inputs_dir: Optional[Path] = None,
    ):
        """
        Initialize a SingleCellDataset instance.

        Args:
            dataset_type_name (str): Name of the dataset type (used for directory naming).
            path (Path): Path to the dataset file.
            organism (Organism): Enum value indicating the organism.
            task_inputs_dir (Optional[Path]): Directory for storing task-specific inputs.
        """
        super().__init__(dataset_type_name, path, organism, task_inputs_dir)

    def load_data(self, backed: Literal["r", "r+"] | bool | None = None) -> None:
        """
        Load the dataset from the path.

        This method reads the dataset file in H5AD format and loads it into the
        `adata` attribute as an AnnData object.

        Args:
            backed (Literal['r', 'r+'] | bool | None): Whether to load the dataset
                into memory or use backed mode.
                Memory: False or None. Default is None.
                Backed: True, 'r' for read-only, 'r+' for read-write

        Populates:
            adata (ad.AnnData): Loaded AnnData object containing gene expression data.
        """
        load_mode = "backed" if backed in {True, "r", "r+"} else "memory"
        logger.info(f"Loading dataset from {self.path} in {load_mode} mode.")
        self.adata = ad.read_h5ad(self.path, backed=backed)

    def _validate(self) -> None:
        """
        Perform dataset-specific validation.

        Validates the following:
        - Gene names must start with the organism prefix (e.g., "ENSG" for human).
        - Gene names must be stored in `adata.var_names` or `adata.var['ensembl_id']`.
        - `adata.X` must contain raw counts (non-negative integers).

        Raises:
            ValueError: If gene names are invalid or `adata.X` does not contain raw counts.
        """
        # Validate gene names
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

        # Attempt to fix gene names if invalid
        if not var:
            if "ensembl_id" in self.adata.var.columns:
                self.adata.var_names = pd.Index(list(self.adata.var["ensembl_id"]))
                var = all(self.adata.var_names.str.startswith(self.organism.prefix))

        # Raise error if gene names are still invalid
        if not var:
            raise ValueError(
                "Dataset does not contain valid gene names. Gene names must"
                f" start with {self.organism.prefix} and be stored in either"
                f" adata.var_names or adata.var['ensembl_id']."
            )
