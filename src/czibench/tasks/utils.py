from typing import List
import scanpy as sc
from anndata import AnnData

# TODO: Later we can add cluster parameters as kwargs here and add them to the task config
def cluster_embedding(adata: AnnData, obsm_key: str = "emb") -> List[int]:
    """Cluster the embedding using the Leiden algorithm"""
    sc.pp.neighbors(adata, use_rep=obsm_key)
    sc.tl.leiden(adata, key_added="leiden", flavor="igraph", n_iterations=2)
    return list(adata.obs["leiden"])
