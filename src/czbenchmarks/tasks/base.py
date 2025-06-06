from abc import ABC, abstractmethod
from typing import List, Optional, Union
import pandas as pd
import anndata as ad

from .constants import RANDOM_SEED
from ..datasets.types import Embedding
from ..metrics.types import MetricResult
from .utils import run_standard_scrna_workflow


class BaseTask(ABC):
    """Abstract base class for all benchmark tasks.

    Defines the interface that all tasks must implement. Tasks are responsible for:
    1. Declaring their required input/output data types
    2. Running task-specific computations
    3. Computing evaluation metrics

    Tasks should store any intermediate results as instance variables
    to be used in metric computation.

    Args:
        random_seed (int): Random seed for reproducibility
    """

    def __init__(
        self,
        *,
        random_seed: int = RANDOM_SEED,
    ):
        self.random_seed = random_seed
        self.requires_multiple_datasets = False

    @abstractmethod
    def _run_task(self, embedding: Embedding, **kwargs) -> dict:
        """Run the task's core computation.

        Should store any intermediate results needed for metric computation
        as instance variables.

        Args:
            embedding: embedding to use for the task
            **kwargs: Additional arguments passed to the task
        Returns:
            Dictionary of output data for the task
        """

    @abstractmethod
    def _compute_metrics(self, **kwargs) -> List[MetricResult]:
        """Compute evaluation metrics for the task.

        Returns:
            List of MetricResult objects containing metric values and metadata
        """

    def _run_task_for_dataset(
        self,
        embedding: Embedding,
        task_kwargs: dict = {},
        metric_kwargs: dict = {},
    ) -> List[MetricResult]:
        """Run task for a dataset or list of datasets and compute metrics.

        This method runs the task implementation and computes the corresponding metrics.

        Args:
            embedding: embedding to use for the task
            task_kwargs: Additional arguments passed to the task
            metric_kwargs: Additional arguments passed to the metrics
        Returns:
            List of MetricResult objects

        """

        task_output = self._run_task(embedding=embedding, **task_kwargs)
        
        # Handle cases where embedding required by metrics but not set by _run_task
        if "embedding" not in metric_kwargs:
            task_output.setdefault("embedding", embedding)
        
        metrics = self._compute_metrics(**task_output, **metric_kwargs)
        return metrics

    def set_baseline(
        self, embedding: Embedding, obs: pd.DataFrame, var: pd.DataFrame, **kwargs
    ) -> Embedding:
        """Set a baseline embedding using PCA on gene expression data.

        This method performs standard preprocessing on the raw gene expression data
        and uses PCA for dimensionality reduction. It then sets the PCA embedding
        as the BASELINE model output in the dataset, which can be used for comparison
        with other model embeddings.

        Args:
            embedding: embedding to use for anndata
            obs: obs dataframe to use for anndata
            var: var dataframe to use for anndata
            **kwargs: Additional arguments passed to run_standard_scrna_workflow
        """

        # Create the AnnData object
        adata = ad.AnnData(X=embedding, obs=obs, var=var)

        # Run the standard preprocessing workflow
        embedding_baseline = run_standard_scrna_workflow(adata, **kwargs)
        return embedding_baseline

    def run(
        self,
        embedding: Union[Embedding, List[Embedding]],
        task_kwargs: Optional[dict] = None,
        metric_kwargs: Optional[dict] = None,
    ) -> Union[List[MetricResult], List[List[MetricResult]]]:
        """Run the task on input data and compute metrics.

        Args:
            embedding: embedding to use for the task
            task_kwargs: Additional arguments passed to the task
            metric_kwargs: Additional arguments passed to the metrics

        Returns:
            For single embedding: A metric result of list of metric results for the task
            For multiple embeddings: List of metric results for each task, one per dataset

        Raises:
            ValueError: If task requires multiple embeddings but single embeddings provided
        """

        # Check if task requires embeddings from multiple datasets
        if self.requires_multiple_datasets and not isinstance(embedding, list):
            raise ValueError("This task requires a list of embeddings")

        # Handle single vs multiple embeddings
        if isinstance(embedding, list):
            # Process each embedding individually
            all_metrics = []
            for emb in embedding:
                all_metrics.append(
                    self._run_task_for_dataset(
                        embedding=emb,
                        task_kwargs=task_kwargs,
                        metric_kwargs=metric_kwargs,
                    )
                )
            return all_metrics
        else:
            # Process single embedding or multiple embeddings as required by the task
            return self._run_task_for_dataset(
                embedding=embedding,
                task_kwargs=task_kwargs,
                metric_kwargs=metric_kwargs,
            )
