"""Simple model for testing"""

from czbenchmarks.models.types import ModelType
from czbenchmarks.datasets.types import DataType
import scanpy as sc


class SimpleModel:
    """A model that generates embeddings on the dataset outputs"""

    def __init__(self):
        self.model_type = ModelType.SCGPT

    def run_inference(self, dataset):
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
        embeddings = adata.obsm['X_pca']
        
        dataset.outputs[self.model_type] = {DataType.EMBEDDING: embeddings}
        return dataset 