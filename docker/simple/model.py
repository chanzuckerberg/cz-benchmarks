import argparse
from typing import Set
from czbenchmarks.datasets.types import DataType, Organism
from czbenchmarks.models.implementations.base_model_implementation import BaseModelImplementation
from czbenchmarks.models.types import ModelType
from czbenchmarks.models.validators.base_single_cell_model_validator import BaseSingleCellValidator
import scanpy as sc

class Simple(BaseModelImplementation, BaseSingleCellValidator):
    def parse_args(self):
        parser = argparse.ArgumentParser(description="Run Simple model on input dataset.")
        parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing the dataset.")
        return parser.parse_args()

    model_type = ModelType.YOUR_MODEL
    available_organisms = [Organism.HUMAN, Organism.MOUSE]  # Use appropriate Organism enums
    required_obs_keys = []  # Specify required obs keys, as needed
    required_var_keys = ["feature_name"]  # Use appropriate feature name

    @property
    def inputs(self) -> Set[DataType]:
        return { DataType.ANNDATA }
    
    @property
    def outputs(self) -> Set[DataType]:
        # Specify appropriate `DataType`s below (embeddings are a typical model output)
        return { DataType.EMBEDDING }  

    def get_model_weights_subdir(self, dataset) -> str:
        return "."

    def _download_model_weights(self, dataset) -> None:
        # Implement your model weight download or verification logic here.
        pass

    def run_model(self, dataset):
        print(self.args)

        """Generate embeddings on the dataset outputs"""
        # Get the raw data
        adata = dataset.adata.copy()

        # Standard preprocessing
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Find highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=1000)
        adata = adata[:, adata.var.highly_variable]

        # Use PCA as embeddings
        sc.pp.scale(adata)
        sc.tl.pca(adata, n_comps=100)
        embeddings = adata.obsm["X_pca"]
        
        dataset.set_output(self.model_type, DataType.EMBEDDING, embeddings)

if __name__ == "__main__":
    Simple().run()
