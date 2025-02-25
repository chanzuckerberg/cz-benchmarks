import anndata as ad
import pandas as pd
from typing import Dict, List
import numpy as np
from .base import BaseDataset
from .types import Organism, DataType


class SingleCellDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        organism: Organism,
    ):
        super().__init__(path)
        self.set_input(DataType.ORGANISM, organism)

    def load_data(self) -> None:
        adata = ad.read_h5ad(self.path)
        self.set_input(DataType.ANNDATA, adata)
        self.set_input(DataType.METADATA, adata.obs)

    def unload_data(self) -> None:
        self._inputs.pop(DataType.ANNDATA, None)
        self._inputs.pop(DataType.METADATA, None)

    @property
    def organism(self) -> Organism:
        return self.get_input(DataType.ORGANISM)

    @property
    def adata(self) -> ad.AnnData:
        return self.get_input(DataType.ANNDATA)

    def _validate(self) -> None:
        if not self.adata:
            raise ValueError("Dataset does not contain anndata object")

        if not self.organism:
            raise ValueError("Organism is not specified")

        if not isinstance(self.organism, Organism):
            raise ValueError("Organism is not a valid Organism enum")

        var = all(self.adata.var_names.str.startswith(self.organism.prefix))

        if not var:
            if "ensembl_id" in self.adata.var.columns:
                self.adata.var_names = pd.Index(list(self.adata.var["ensembl_id"]))
                var = all(self.adata.var_names.str.startswith(self.organism.prefix))

        if not var:
            raise ValueError(
                "Dataset does not contain valid gene names. Gene names must"
                f" start with {self.organism.prefix}"
            )


class PerturbationSingleCellDataset(SingleCellDataset):
    def __init__(
        self,
        path: str,
        organism: Organism,
        condition_key: str = "condition",
    ):
        super().__init__(path, organism)
        self.set_input(DataType.CONDITION_KEY, condition_key)

    def load_data(self) -> None:
        super().load_data()
        assert (
            self.condition_key in self.adata.obs.columns
        ), f"Condition key {self.condition_key} not found in adata.obs"

        # Store control data for each condition in the reference dataset
        conditions = np.array(list(self.adata.obs[self.condition_key]))
        truth_data = {
            condition: pd.DataFrame(
                data=self.adata[conditions == condition].X.toarray(),
                index=self.adata[conditions == condition].obs_names,
                columns=self.adata[conditions == condition].var_names,
            )
            for condition in set(conditions)
        }

        self.set_output(
            DataType.PERTURBATION_TRUTH,
            truth_data,
        )

    @property
    def perturbation_truth(self) -> Dict[str, pd.Series]:
        return self.get_input(DataType.PERTURBATION_TRUTH)

    @property
    def condition_key(self) -> str:
        return self.get_input(DataType.CONDITION_KEY)

    @property
    def available_perturbations(self) -> List[str]:
        return list(self.perturbation_truth.keys())
