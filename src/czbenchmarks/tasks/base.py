from abc import ABC, abstractmethod
from typing import List, Optional, Union
import pandas as pd
import anndata as ad

from ..constants import RANDOM_SEED
from ..datasets.types import Embedding, GeneExpression
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
        # FIXME should this be changed to requires_multiple_embeddings?
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
        self,
        expression_data: GeneExpression,
        obs: Optional[pd.DataFrame] = None,
        var: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> Embedding:
        """Set a baseline embedding using PCA on gene expression data.

        This method performs standard preprocessing on the raw gene expression data
        and uses PCA for dimensionality reduction. It then sets the PCA embedding
        as the BASELINE model output in the dataset, which can be used for comparison
        with other model embeddings.

        Args:
            expression_data: expression data to use for anndata
            obs: obs dataframe to use for anndata
            var: var dataframe to use for anndata
            **kwargs: Additional arguments passed to run_standard_scrna_workflow
        """

        # Create the AnnData object
        # FIXME MICHELLE -- obs var probably not needed, but needs debugging
        adata = ad.AnnData(X=expression_data)#, obs=obs, var=var)

        # Run the standard preprocessing workflow
        embedding_baseline = run_standard_scrna_workflow(adata, **kwargs)
        return embedding_baseline

    def run(
        self,
        embedding: Union[Embedding, List[Embedding]],
        task_kwargs: dict = {},
        metric_kwargs: dict = {},
    ) -> List[MetricResult]:
        """Run the task on input data and compute metrics.

        Args:
            embedding: embedding to use for the task
            task_kwargs: Additional arguments passed to the task
            metric_kwargs: Additional arguments passed to the metrics

        Returns:
            For single embedding: A metric result of list of metric results for the task
            For multiple embeddings: List of metric results for each task, one per dataset

        Raises:
            ValueError: If input does not match multiple embedding requirement
        """

        # Check if task requires embeddings from multiple datasets
        if self.requires_multiple_datasets:
            error_message = "This task requires a list of embeddings"
            if not isinstance(embedding, list):
                raise ValueError(error_message)
            if not all([isinstance(emb, Embedding) for emb in embedding]):
                raise ValueError(error_message)
            if len(embedding) < 2:
                raise ValueError(f"{error_message} but only one embedding provided")
        else:
            if not isinstance(embedding, Embedding):
                raise ValueError("This task requires a single embedding for input")

        return self._run_task_for_dataset(
                                    embedding=embedding,
                                    task_kwargs=task_kwargs,
                                    metric_kwargs=metric_kwargs,
                                )