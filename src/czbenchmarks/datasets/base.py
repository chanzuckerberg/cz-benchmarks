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
    def unload_data(self) -> None:
        """
        Unload the dataset from memory.

        This method should be implemented by subclasses to free memory by
        clearing loaded data.
        For example, SingleCellDataset sets its AnnData object to None.

        This is used to clear memory-intensive data before serialization,
        since serializing large raw data artifacts can be error-prone and
        inefficient.

        Any instance attributes containing loaded data should be cleared or
        set to None.
        """
