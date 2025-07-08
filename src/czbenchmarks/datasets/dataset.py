from abc import ABC, abstractmethod
from pathlib import Path
from typing import IO, Any
import os

from .types import Organism


class Dataset(ABC):
    """
    A base class for task-specific datasets. Each concrete dataset class will extract the data required for a specific type of task from the provided input file. These task specific data items can then be retrieved as object instance variables or written to files for later use.
    
    path: Path

    Args:
        ABC (_type_): _description_
    """
    
    name: str # TODO: rename to dataset_type?
    path: Path
    dir: Path
    organism: Organism
    
    
    def __init__(self, name: str, path: Path, organism: Organism, **kwargs: Any):
        self.name = name
        
        self.path = path
        if not os.path.exists(self.path):
            raise ValueError("Dataset path does not exist")
        
        self.dir = path.parent
        if not self.dir.is_dir():
            raise ValueError(f"Dataset path {self.path} is not a directory")

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
    def store_task_inputs(self) -> Path:
        """
        Store the task-specific inputs that have been extracted from the dataset. These files should be stored under the dataset path in a subdirectory who name is keyed to the subclass.

        This method should be implemented by subclasses.
        
        Returns:
            Path: The path to the directory storing the task input files.
        """
        pass

    def _store_task_input(self, filename: str, data: IO) -> Path:
        """ Store a task input data to a file in a subdirectory of the dataset directory, named after the dataset type."""
        output_dir = Path(self.dir) / self.name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / filename
        with open(output_file, 'w') as f:
            f.write(data)
        return output_file

    @abstractmethod
    def _validate(self) -> None:
        pass

    # FIXME VALIDATION: move to validation class?
    def validate(self) -> None:
        """Validate that all inputs and outputs match their expected types"""

        if not isinstance(self.organism, Organism):
            raise ValueError("Organism is not a valid Organism enum")

        self._validate()


