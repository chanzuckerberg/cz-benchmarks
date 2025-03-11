import logging
from typing import Set, List
import scanpy as sc

from ..datasets import BaseDataset, DataType
from ..metrics import MetricType, metrics
from ..metrics.types import MetricResult
from ..models.types import ModelType
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

    def _run_task(self, data: BaseDataset, model_type: ModelType):
        """Runs clustering on the embedding data.

        Performs clustering and stores results for metric computation.

        Args:
            data: Dataset containing embedding and ground truth labels
        """
        # Get anndata object and add embedding
        adata = data.adata
        adata.obsm["emb"] = data.get_output(model_type, DataType.EMBEDDING)

        # Store labels and generate clusters
        self.input_labels = data.get_input(DataType.METADATA)[self.label_key]
        self.predicted_labels = cluster_embedding(adata, obsm_key="emb")

    def _compute_metrics(self) -> List[MetricResult]:
        """Computes clustering evaluation metrics.

        Returns:
            List of MetricResult objects containing ARI and NMI scores
        """
        return [
            MetricResult(
                metric_type=metric_type,
                value=metrics.compute(
                    metric_type,
                    labels_true=self.input_labels,
                    labels_pred=self.predicted_labels,
                ),
            )
            for metric_type in [
                MetricType.ADJUSTED_RAND_INDEX,
                MetricType.NORMALIZED_MUTUAL_INFO,
            ]
        ]

    def run_baseline(
        self, data: BaseDataset, n_top_genes=3000, n_pcs=50, random_state=42
    ):
        """Run a baseline clustering using PCA on gene expression.

        Instead of using embeddings from a model, this method performs standard
        preprocessing on the raw gene expression data and uses PCA for dimensionality
        reduction before clustering. This provides a baseline performance to compare
        against model-generated embeddings.

        Args:
            data: SingleCellDataset containing AnnData with gene expression and metadata
            n_top_genes: Number of highly variable genes to select
            n_pcs: Number of principal components to use
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing baseline clustering metrics
        """

        # Get the AnnData object from the dataset
        adata = data.get_input(DataType.ANNDATA)

        # Standard preprocessing steps for single-cell data
        sc.pp.normalize_total(adata)  # Normalize counts per cell
        sc.pp.log1p(adata)  # Log-transform the data

        # Identify highly variable genes using Seurat v3 method
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat_v3")

        # Subset to only highly variable genes to reduce noise
        adata = adata[:, adata.var["highly_variable"]]

        # Run PCA for dimensionality reduction
        sc.pp.pca(adata, n_comps=n_pcs, random_state=random_state)

        # Use PCA result as the embedding for clustering
        data.set_output(ModelType.BASELINE, DataType.EMBEDDING, adata.obsm["X_pca"])

        # Run the clustering task with the PCA embedding
        baseline_metrics = self.run(data, model_types=[ModelType.BASELINE])[
            ModelType.BASELINE
        ]

        return baseline_metrics
