import logging
import os
from abc import ABC, abstractmethod
from typing import ClassVar, Type, Set, List
import glob
import pathlib

from ..constants import (
    INPUT_DATA_PATH_DOCKER,
    MODEL_WEIGHTS_PATH_DOCKER,
    OUTPUT_DATA_PATH_DOCKER,
    get_numbered_path,
    get_base_name
)
from ..datasets.base import BaseDataset
from ..datasets.types import DataType, DataValue

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # This ensures the configuration is applied
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseModelValidator(ABC):
    # Type annotation for class variable
    dataset_type: ClassVar[Type[BaseDataset]]

    def __init_subclass__(cls) -> None:
        """Validate that subclasses define required class variables"""
        super().__init_subclass__()

        if cls.__name__ != "BaseModelImplementation":
            if not hasattr(cls, "dataset_type"):
                raise TypeError(
                    f"Can't instantiate {cls.__name__}"
                    " without dataset_type class variable"
                )

    @abstractmethod
    def _validate_dataset(self, dataset: BaseDataset):
        pass

    @property
    @abstractmethod
    def inputs(self) -> Set[DataType]:
        """Specify what input types this model requires"""

    @property
    @abstractmethod
    def outputs(self) -> Set[DataType]:
        """Specify what output types this model produces"""

    def validate_dataset(self, dataset: BaseDataset):
        if not isinstance(dataset, self.dataset_type):
            raise ValueError("Dataset type mismatch")

        # Validate required inputs are available
        missing_inputs = self.inputs - set(dataset.inputs.keys())
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")

        self._validate_dataset(dataset)


class BaseModelImplementation(BaseModelValidator, ABC):
    datasets: List[BaseDataset]
    model_weights_dir: str

    @abstractmethod
    def get_model_weights_subdir(self, dataset: BaseDataset) -> str:
        """Return the subdirectory (if applicable) where this model variant's
        weights should be stored.

        If the model variant does not require a subdirectory, return an empty string.
        """

    @abstractmethod
    def _download_model_weights(self, dataset: BaseDataset):
        pass

    def download_model_weights(self, dataset: BaseDataset) -> None:
        self.model_weights_dir = (
            f"{MODEL_WEIGHTS_PATH_DOCKER}/{self.get_model_weights_subdir(dataset)}"
        )

        if not os.path.exists(self.model_weights_dir) or not any(
            os.listdir(self.model_weights_dir)
        ):
            logger.info("Downloading model weights...")
            self._download_model_weights(dataset)
            logger.info("Model weights downloaded successfully")
        else:
            logger.info("Model weights already downloaded...")

    @abstractmethod
    def run_model(self) -> None:
        """Implement model-specific inference logic"""

    @abstractmethod
    def parse_args(self):
        """Return parsed arguments for the model."""

    def run(self):
        # Find all input datasets
        input_dir = pathlib.Path(INPUT_DATA_PATH_DOCKER).parent
        base_pattern = get_base_name(INPUT_DATA_PATH_DOCKER)
        input_files = sorted(glob.glob(os.path.join(input_dir, base_pattern)))
        
        if not input_files:
            raise FileNotFoundError("No input datasets found")

        # Ensure base file comes first if it exists
        if INPUT_DATA_PATH_DOCKER in input_files:
            input_files.remove(INPUT_DATA_PATH_DOCKER)
            input_files = [INPUT_DATA_PATH_DOCKER] + input_files

        # Load all datasets
        self.datasets = [
            self.dataset_type.deserialize(input_file)
            for input_file in input_files
        ]

        logger.info("Loading data...")
        for dataset in self.datasets:
            dataset.load_data()
        logger.info("Data loaded successfully")

        logger.info("Validating data...")
        for dataset in self.datasets:
            dataset.validate()
            self.validate_dataset(dataset)
        logger.info("Data validated successfully")

        for dataset in self.datasets:
            self.download_model_weights(dataset)

        logger.info("Running model...")
        for dataset in self.datasets:
            self.run_model(dataset)
        logger.info("Model ran successfully")

        # Unload and serialize all datasets
        for i, dataset in enumerate(self.datasets):
            dataset.unload_data()
            output_path = get_numbered_path(OUTPUT_DATA_PATH_DOCKER, None if i == 0 else i)
            dataset.serialize(output_path)