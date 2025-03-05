import logging
from typing import Dict, Set

from ..datasets.single_cell import SingleCellDataset
from ..datasets.types import DataType
from ..metrics import MetricType, metrics
from .base import BaseTask
from .utils import cluster_embedding

logger = logging.getLogger(__name__)


class ClusteringTask(BaseTask):
    """Task for evaluating clustering performance against ground truth labels.

    This task performs clustering on embeddings and evaluates the results
    using multiple clustering metrics (ARI and NMI).

    Args:
        label_key (str): Key to access ground truth labels in metadata
    """

    def __init__(self, label_key: str):
        self.label_key = label_key

    @property
    def required_inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set of required input DataTypes (metadata with labels)
        """
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        """Required output data types.

        Returns:
            required output types from models this task to run (embedding to cluster)
        """
        return {DataType.EMBEDDING}

    def _run_task(self, data: SingleCellDataset):
        """Runs clustering on the embedding data.

        Performs clustering and stores results for metric computation.

        Args:
            data: Dataset containing embedding and ground truth labels
        """
        # Get anndata object and add embedding
        adata = data.adata
        adata.obsm["emb"] = data.get_output(DataType.EMBEDDING)

        # Store labels and generate clusters
        self.input_labels = data.get_input(DataType.METADATA)[self.label_key]
        self.predicted_labels = cluster_embedding(adata, obsm_key="emb")

    def _compute_metrics(self) -> Dict[str, float]:
        """Computes clustering evaluation metrics.

        Returns:
            Dictionary containing ARI and NMI scores
        """
        return {
            metric_type.value: metrics.compute(
                metric_type,
                labels_true=self.input_labels,
                labels_pred=self.predicted_labels,
            )
            for metric_type in [
                MetricType.ADJUSTED_RAND_INDEX,
                MetricType.NORMALIZED_MUTUAL_INFO,
            ]
        }

    def run_baseline(
        self,
        data: SingleCellDataset,
        min_genes=200,
        min_cells=3,
        target_sum=1e4,
        min_mean=0.0125,
        max_mean=3,
        min_disp=0.5,
        n_pcs=50,
        resolution=0.5,
        random_state=42,
    ):
        import scanpy as sc

        adata = data.get_input(DataType.METADATA)
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        sc.pp.normalize_total(adata, target_sum=target_sum)
        hvg_params = dict(min_mean=min_mean, max_mean=max_mean, min_disp=min_disp)
        sc.pp.highly_variable_genes(adata, **hvg_params)
        adata = adata[:, adata.var["highly_variable"]]
        sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)
        sc.pp.neighbors(adata, use_rep="X_pca")
        sc.tl.leiden(adata, resolution=resolution, random_state=random_state)
