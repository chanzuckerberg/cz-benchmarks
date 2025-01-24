import anndata as ad
import pandas as pd
from typing import Optional, Any
from .base import BaseDataset
from .types import Organism

class SingleCellDataset(BaseDataset):
    def __init__(
            self,
            path: str,
            organism: Organism,
            **kwargs: Any
        ):
        
        self.adata = ad.read_h5ad(path)
        self.sample_metadata = self.adata.obs
        self.organism = organism
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        super().__init__(path)

    def _validate(self) -> None:
        if not hasattr(self, 'adata'):
            raise ValueError("Dataset does not contain anndata object")
        
        if not isinstance(self.organism, Organism):
            raise ValueError("Organism is not a valid Organism enum")
        
        var = all(self.adata.var_names.str.startswith(self.organism.prefix))
        
        if not var:
            if 'ensembl_id' in self.adata.var.columns:
                self.adata.var_names = pd.Index(self.adata.var['ensembl_id'].values)
                # TODO check if var names satisfies the prefix criteria
            else:
                raise ValueError(f"Dataset does not contain valid gene names. Gene names must start with {self.organism.prefix}")
