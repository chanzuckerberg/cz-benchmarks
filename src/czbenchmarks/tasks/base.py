from abc import ABC, abstractmethod
from typing import List, Optional, Union
import numpy as np

from .constants import RANDOM_SEED
from ..datasets import BaseDataset
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
    def _run_task(self, data: BaseDataset, embedding: np.ndarray) -> dict:
        """Run the task's core computation.

        Should store any intermediate results needed for metric computation
        as instance variables.

        Args:
            data: Dataset containing required input and output data
            embedding: embedding to use for the task
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
        data: BaseDataset,
        embedding: Optional[np.ndarray] = None,
    ) -> List[MetricResult]:
        """Run task for a dataset or list of datasets and compute metrics.

        This method runs the task implementation and computes the corresponding metrics.

        Args:
            data: dataset to run the task on
            embedding: embedding to use for the task, if required. Default is None.
        Returns:
            List of MetricResult objects

        """
        task_output = self._run_task(data=data, embedding=embedding)
        # Handle cases where embedding required by metrics but not set by _run_task
        if embedding:
            task_output.setdefault("embedding", embedding)
        metrics = self._compute_metrics(**task_output)
        return metrics

    def set_baseline(self, data: BaseDataset, **kwargs) -> np.ndarray:
        """Set a baseline embedding using PCA on gene expression data.

        This method performs standard preprocessing on the raw gene expression data
        and uses PCA for dimensionality reduction. It then sets the PCA embedding
        as the BASELINE model output in the dataset, which can be used for comparison
        with other model embeddings.

        Args:
            data: BaseDataset containing AnnData with gene expression data
            **kwargs: Additional arguments passed to run_standard_scrna_workflow
        """

        # FIXME BYODATASET: decouple AnnData
        # Get the AnnData object from the dataset
        adata = data.adata

        # Run the standard preprocessing workflow
        embedding_baseline = run_standard_scrna_workflow(adata, **kwargs)
        return embedding_baseline

    def run(
        self,
        data: Union[BaseDataset, List[BaseDataset]],
        embedding: Optional[np.ndarray] = None,
        **kwargs,
    ) -> Union[List[MetricResult], List[List[MetricResult]]]:
        """Run the task on input data and compute metrics.

        Args:
            data: Single dataset or list of datasets to evaluate. Must contain
                required input and output data types.
            embedding: embedding to use for the task
            **kwargs: Additional arguments passed to the task

        Returns:
            For single dataset: A metric result of list of metric results for the task
            For multiple datasets: List of metric results for each task, one per dataset

        Raises:
            ValueError: If task requires multiple datasets but single dataset provided
        """

        # Check if task requires multiple datasets
        if self.requires_multiple_datasets and not isinstance(data, list):
            raise ValueError("This task requires a list of datasets")

        # Handle single vs multiple datasets
        if isinstance(data, list):
            # Process each dataset individually
            all_metrics = []
            for d in data:
                all_metrics.append(
                    self._run_task_for_dataset(data=d, embedding=embedding, **kwargs)
                )
            return all_metrics
        else:
            # Process single dataset or multiple datasets as required by the task
            return self._run_task_for_dataset(data=data, embedding=embedding, **kwargs)
