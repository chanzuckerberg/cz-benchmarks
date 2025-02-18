import os
from abc import ABC, abstractmethod
from typing import Any, Dict
import dill

from ..datasets.types import DataType, DataValue


class BaseDataset(ABC):
    inputs: Dict[DataType, DataValue] = {}
    outputs: Dict[DataType, DataValue] = {}

    def __init__(self, path: str, **kwargs: Any):
        self.path = path
        self.kwargs = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def _validate(self) -> None:
        pass

    def validate(self) -> None:
        if not os.path.exists(self.path):
            raise ValueError(f"Dataset {self.path} is not valid")

        """Validate that all inputs and outputs match their expected types"""
        for data_type, value in self.inputs.items():
            if not isinstance(value, data_type.dtype):
                raise TypeError(
                    f"Input {data_type.name} has incorrect type: "
                    f"expected {data_type.dtype}, got {type(value)}"
                )

        for data_type, value in self.outputs.items():
            if not isinstance(value, data_type.dtype):
                raise TypeError(
                    f"Output {data_type.name} has incorrect type: "
                    f"expected {data_type.dtype}, got {type(value)}"
                )

        self._validate()

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

    def serialize(self, path: str) -> None:
        """
        Serialize this dataset instance to disk using dill.

        Args:
            path: Path where the serialized dataset should be saved
        """
        if not path.endswith(".dill"):
            path = f"{path}.dill"

        with open(path, "wb") as f:
            dill.dump(self, f)

    @staticmethod
    def deserialize(path: str) -> "BaseDataset":
        """
        Load a serialized dataset from disk.

        Args:
            path: Path to the serialized dataset file

        Returns:
            BaseDataset: The deserialized dataset instance
        """
        if not path.endswith(".dill"):
            path = f"{path}.dill"

        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found at {path}")

        with open(path, "rb") as f:
            return dill.load(f)
