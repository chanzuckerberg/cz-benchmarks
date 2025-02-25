import anndata as ad
import pandas as pd
from .base import BaseDataset
from .types import Organism, DataType


class SingleCellDataset(BaseDataset):
    def __init__(
        self,
        source_path: str,
        organism: Organism,
    ):
        super().__init__(source_path)
        self.set_input(DataType.ORGANISM, organism)

    def load_data(self) -> None:
        adata = ad.read_h5ad(self.local_path)
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
