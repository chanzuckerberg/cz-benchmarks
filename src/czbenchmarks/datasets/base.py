from abc import ABC, abstractmethod
from typing import Any

from .types import Organism


class BaseDataset(ABC):
    def __init__(self, path: str, organism: Organism, **kwargs: Any):
        self.path = path
        self.organism = organism
        self.kwargs = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def load_data(self) -> None:
        """
        Load the dataset into memory.

        This method should be implemented by subclasses to load their specific
        data format.
        For example, SingleCellDataset loads an AnnData object from an h5ad
        file.

        The loaded data should be stored as instance attributes that can be
        accessed by other methods.
        """

    @abstractmethod
    def cache_data(self) -> None:
        """
        Cache existing data.

        This method should be implemented by subclasses to cache existing data.
        For example, after extracting dataset embeddings from a model,
        the anndata object with the embeddings in the obsm slot can be cached.
        """
