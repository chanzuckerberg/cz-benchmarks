from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path
from typing import Any, Optional

from .types import Organism


class Dataset(ABC):
    """
    Abstract base class for datasets.

    Each concrete Dataset subclass is responsible for extracting and managing the data required for a specific
    type of task from the provided input file. Subclasses should define instance variables to store these
    task-specific data items, which can then be accessed as object attributes or written to files for downstream use.

    All Dataset instances must specify an `Organism` enum value to indicate the organism from which the data was derived.

    Subclasses must implement:
        - `load_data`: Loads the dataset from the input file and populates relevant instance variables.
        - `store_task_inputs`: Stores the extracted task-specific inputs in files or directories as needed.
        - `_validate`: Validates dataset-specific constraints and requirements.

    Attributes:
        path: The path to the dataset file.
        task_inputs_dir: The directory where task-specific input files are stored.
        organism: The organism from which the data was derived.
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
        """
        Initialize a Dataset instance.

        Args:
            dataset_type_name (str): Name of the dataset type (used for directory naming).
            path (str | Path): Path to the dataset file.
            organism (Organism): Enum value indicating the organism.
            task_inputs_dir (Optional[Path]): Directory for storing task-specific inputs.
            kwargs (Any): Additional attributes for the dataset.

        Raises:
            ValueError: If the dataset path does not exist.
        """
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
        Load the dataset from its source file into memory.

        Subclasses must implement this method to load their specific data format.
        For example, SingleCellDataset loads an AnnData object from an h5ad file.

        The loaded data should be stored as instance attributes that can be
        accessed by other methods.
        """

    @abstractmethod
    def store_task_inputs(self) -> Path:
        """
        Store the task-specific inputs extracted from the dataset.

        Subclasses must implement this method to store task-specific files in a
        subdirectory of the dataset path. The subdirectory name is determined by
        the subclass.

        Returns:
            Path: The path to the directory storing the task input files.
        """
        pass

    def _store_task_input(self, path: Path | str, data: StringIO) -> None:
        """
        Store a single task input data stream to a file.
        Creates the necessary subdirectories if they do not exist.

        Args:
            path (Path | str): Relative path to the task input file.
            data (StringIO): Data to write to the file.
        """
        output_dir = self.task_inputs_dir / Path(path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.task_inputs_dir / path
        output_file.write_text(data)

    @abstractmethod
    def _validate(self) -> None:
        """
        Perform dataset-specific validation.

        Subclasses must implement this method to validate dataset-specific constraints.
        """
        pass

    # FIXME VALIDATION: move to validation class?
    def validate(self) -> None:
        """
        Performs general validation checks, such as ensuring the organism is a valid
        `Organism` enum value. Calls `_validate` for subclass-specific validation.

        Raises:
            ValueError: If validation fails.
        """

        if not isinstance(self.organism, Organism):
            raise ValueError("Organism is not a valid Organism enum")

        self._validate()
