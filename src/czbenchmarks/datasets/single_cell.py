import anndata as ad
import pandas as pd
from typing import Dict
import numpy as np
from .base import BaseDataset
from .types import Organism, DataType
import logging

logger = logging.getLogger(__name__)


class SingleCellDataset(BaseDataset):
    """Single cell dataset containing gene expression data and metadata.

    Handles loading and validation of AnnData objects with gene expression data
    and associated metadata for a specific organism."""

    def __init__(
        self,
        path: str,
        organism: Organism,
    ):
        super().__init__(path, organism)

    def load_data(self) -> None:
        adata = ad.read_h5ad(self.path)
        self.set_input(DataType.ANNDATA, adata)
        self.set_input(DataType.METADATA, adata.obs)

    def unload_data(self) -> None:
        self._inputs.pop(DataType.ANNDATA, None)
        self._inputs.pop(DataType.METADATA, None)

    @property
    def adata(self) -> ad.AnnData:
        return self.get_input(DataType.ANNDATA)

    def _validate(self) -> None:
        if DataType.ANNDATA not in self._inputs:
            raise ValueError("Dataset does not contain anndata object")

        if DataType.ORGANISM not in self._inputs:
            raise ValueError("Organism is not specified")

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


class PerturbationSingleCellDataset(SingleCellDataset):
    """
    Single cell dataset with perturbation data, containing control and
    perturbed cells.

    Input data requirements:

    - H5AD file containing single cell gene expression data
    - Must have a condition column in adata.obs specifying control ("ctrl") and
      perturbed conditions.
    - Must have a split column in adata.obs to identify test samples
    - Condition format must be one of:

      - ``ctrl`` for control samples
      - ``{gene}+ctrl`` for single gene perturbations
      - ``{gene1}+{gene2}`` for combinatorial perturbations
    """

    def __init__(
        self,
        path: str,
        organism: Organism,
        condition_key: str = "condition",
        split_key: str = "split",
    ):
        super().__init__(path, organism)
        self.set_input(DataType.CONDITION_KEY, condition_key)
        self.set_input(DataType.SPLIT_KEY, split_key)

    def load_data(self) -> None:
        super().load_data()
        if self.condition_key not in self.adata.obs.columns:
            raise ValueError(
                f"Condition key {self.condition_key} not found in adata.obs"
            )
        if self.split_key not in self.adata.obs.columns:
            raise ValueError(f"Split key {self.split_key} not found in adata.obs")

        # Store control data for each condition in the reference dataset
        conditions = np.array(list(self.adata.obs[self.condition_key]))

        test_conditions = set(
            self.adata.obs[self.condition_key][self.adata.obs[self.split_key] == "test"]
        )

        truth_data = {
            str(condition): pd.DataFrame(
                data=self.adata[conditions == condition].X.toarray(),
                index=self.adata[conditions == condition].obs_names,
                columns=self.adata[conditions == condition].var_names,
            )
            for condition in set(test_conditions)
        }

        self.set_input(
            # This only contains the test conditions, not the training conditions
            DataType.PERTURBATION_TRUTH,
            truth_data,
        )

        self.set_input(
            DataType.ANNDATA,
            self.adata[self.adata.obs[self.condition_key] == "ctrl"].copy(),
        )

    def unload_data(self) -> None:
        super().unload_data()
        self._inputs.pop(DataType.PERTURBATION_TRUTH, None)

    @property
    def perturbation_truth(self) -> Dict[str, pd.DataFrame]:
        return self.get_input(DataType.PERTURBATION_TRUTH)

    @property
    def condition_key(self) -> str:
        return self.get_input(DataType.CONDITION_KEY)

    @property
    def split_key(self) -> str:
        return self.get_input(DataType.SPLIT_KEY)

    def _validate(self) -> None:
        super()._validate()

        # Validate split values
        valid_splits = {"train", "test", "val"}
        splits = set(self.adata.obs[self.split_key])
        invalid_splits = splits - valid_splits
        if invalid_splits:
            raise ValueError(f"Invalid split value(s): {invalid_splits}")

        # Validate condition format
        conditions = set(
            list(self.adata.obs[self.condition_key])
            + list(self.perturbation_truth.keys())
        )

        for condition in conditions:
            if condition == "ctrl":
                continue

            parts = condition.split("+")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid perturbation condition format: {condition}. "
                    "Must be 'ctrl', '{gene}+ctrl', or '{gene1}+{gene2}'"
                )
