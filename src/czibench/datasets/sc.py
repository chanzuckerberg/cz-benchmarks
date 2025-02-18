import anndata as ad
import pandas as pd
from .base import BaseDataset
from .types import Organism, DataType


class SingleCellDataset(BaseDataset):
    def __init__(
        self,
        path: str,
        organism: Organism,
    ):
        super().__init__(path)
        self.inputs[DataType.ORGANISM] = organism

    def load_data(self) -> None:
        self.inputs[DataType.ANNDATA] = ad.read_h5ad(self.path)
        self.inputs[DataType.METADATA] = self.inputs[DataType.ANNDATA].obs

    def unload_data(self) -> None:
        self.inputs.pop(DataType.ANNDATA)
        self.inputs.pop(DataType.METADATA)

    @property
    def organism(self) -> Organism:
        return self.inputs.get(DataType.ORGANISM)

    @property
    def adata(self) -> ad.AnnData:
        return self.inputs.get(DataType.ANNDATA)

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
