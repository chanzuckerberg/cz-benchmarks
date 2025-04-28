"""Simple model implementation for testing purposes.

This module contains a simple model that generates random embeddings.
It's useful for testing the framework's functionality without requiring
real model inference or Docker containers.
"""

from czbenchmarks.models.types import ModelType
from czbenchmarks.datasets.types import DataType
import numpy as np


class SimpleModel:
    """A simple model that generates random embeddings.

    This is a dummy implementation that doesn't do any real model inference.
    It just generates random embeddings of the correct shape for testing purposes.
    """

    def __init__(self):
        self.model_type = ModelType.SCGPT

    def run_inference(self, dataset):
        """Generate random embeddings for the dataset.

        Args:
            dataset: The dataset to generate embeddings for

        Returns:
            The dataset with random embeddings added to its outputs
        """
        mock_processed_data = dataset
        n_cells = dataset.adata.n_obs
        dummy_embeddings = np.random.normal(size=(n_cells, 100))
        model_type = self.model_type
        mock_processed_data.outputs[model_type] = {DataType.EMBEDDING: dummy_embeddings}
        return mock_processed_data 