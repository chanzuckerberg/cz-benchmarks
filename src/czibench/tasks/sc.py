from typing import Dict
from .base import BaseTask
from ..metrics.clustering import adjusted_rand_index, normalized_mutual_info
from ..metrics.embedding import silhouette_score
from .utils import cluster_embedding
from ..datasets.sc import SingleCellDataset

class ClusteringTask(BaseTask):
        
    def __init__(self, label_key: str):
        self.label_key = label_key
        
    def validate(self, data: SingleCellDataset):
        return data.output_embedding is not None and self.label_key in data.sample_metadata.columns
        
    def _run_task(self, data: SingleCellDataset) -> SingleCellDataset:
        adata = data.adata
        adata.obsm["emb"] = data.output_embedding
        self.input_labels = data.sample_metadata[self.label_key]
        self.predicted_labels = cluster_embedding(adata, obsm_key = "emb")
        return data
    
    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "adjusted_rand_index": adjusted_rand_index(self.input_labels, self.predicted_labels),
            "normalized_mutual_info": normalized_mutual_info(self.input_labels, self.predicted_labels),
        }

class EmbeddingTask(BaseTask):    
    def __init__(self, label_key: str):
        self.label_key = label_key
        
    def validate(self, data: SingleCellDataset):
        return data.output_embedding is not None and self.label_key in data.sample_metadata.columns
        
    def _run_task(self, data: SingleCellDataset) -> SingleCellDataset:
        # passthrough, embedding already exists
        self.embedding = data.output_embedding
        self.input_labels = data.sample_metadata[self.label_key]
        return data  
    
    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "silhouette_score": silhouette_score(self.embedding, self.input_labels)
        }