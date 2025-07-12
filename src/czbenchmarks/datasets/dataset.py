from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Any, Optional

from .types import Organism


class Dataset(ABC):
    """
    A base class for task-specific datasets. Each concrete Dataset class will extract the data required for a specific
    type of task from the provided input file and store in subclass-determined instance variables. These task specific
    data items can then be retrieved as object instance variables or written to files for later use. All Dataset objects
    specify an `Organism` enum value to indicate the organism from which the data was derived. Each concrete Dataset
    class should implement the `load_data` method to load the data from the input file, and the `store_task_inputs`
    method to store the task-specific inputs that have been extracted from the dataset.
    """

    path: Path
    task_inputs_dir: Path
    organism: Organism

    def __init__(
        self,
        dataset_type_name: str,
        path: str | Path,
        organism: Organism,
        task_inputs_dir: Optional[Path] = None,
        **kwargs: Any,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise ValueError("Dataset path does not exist")

        self.task_inputs_dir = task_inputs_dir or (
            Path(f"{self.path.with_suffix('')}_task_inputs") / dataset_type_name.lower()
        )

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

    def _store_task_input(self, path: Path | str, data: StringIO) -> None:
        """
        Store a task input data to a file in a subdirectory of the dataset directory, named after the dataset type.
        """
        output_dir = self.task_inputs_dir / Path(path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.task_inputs_dir / path
        output_file.write_text(data)

    @abstractmethod
    def _validate(self) -> None:
        pass

    # FIXME VALIDATION: move to validation class?
    def validate(self) -> None:
        """Validate that all inputs and outputs match their expected types"""

        if not isinstance(self.organism, Organism):
            raise ValueError("Organism is not a valid Organism enum")

        self._validate()
