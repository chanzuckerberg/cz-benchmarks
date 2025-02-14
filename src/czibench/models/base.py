import logging
import os
from abc import ABC, abstractmethod
from typing import ClassVar, Type

from ..constants import (
    INPUT_DATA_PATH_DOCKER,
    MODEL_WEIGHTS_PATH_DOCKER,
    OUTPUT_DATA_PATH_DOCKER,
)
from ..datasets.base import BaseDataset

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
        
        if cls.__name__ != 'BaseModelImplementation':
            if not hasattr(cls, "dataset_type"):
                raise TypeError(
                    f"Can't instantiate {cls.__name__} without dataset_type class variable"
                )

    @abstractmethod
    def _validate_dataset(self, dataset: BaseDataset):
        pass

    def validate_dataset(self, dataset: BaseDataset):
        if not isinstance(dataset, self.dataset_type):
            raise ValueError(
                f"Dataset type mismatch: expected {self.dataset_type.__name__}, "
                f"got {type(dataset).__name__}"
            )
        self._validate_dataset(dataset)


class BaseModelImplementation(BaseModelValidator, ABC):
    data: BaseDataset
    model_weights_dir: str
    
    @abstractmethod
    def get_model_weights_subdir(self) -> str:
        """Return the subdirectory (if applicable) where this model variant's
        weights should be stored.

        If the model variant does not require a subdirectory, return an empty string.
        """

    @abstractmethod
    def _download_model_weights(self):
        pass

    def download_model_weights(self) -> None:
        self.model_weights_dir = (
            f"{MODEL_WEIGHTS_PATH_DOCKER}/{self.get_model_weights_subdir()}"
        )

        if not os.path.exists(self.model_weights_dir) or not any(
            os.listdir(self.model_weights_dir)
        ):
            logger.info("Downloading model weights...")
            self._download_model_weights()
            logger.info("Model weights downloaded successfully")
        else:
            logger.info("Model weights already downloaded...")

    @abstractmethod
    def run_model(self) -> None:
        """Implement model-specific inference logic"""

    def run(self):
        self.data = self.dataset_type.deserialize(INPUT_DATA_PATH_DOCKER)

        logger.info("Loading data...")
        self.data.load_data()
        logger.info("Data loaded successfully")

        logger.info("Validating data...")
        self.data.validate()
        self.validate_dataset(self.data)
        logger.info("Data validated successfully")

        self.download_model_weights()

        logger.info("Running model...")
        self.run_model()
        logger.info("Model ran successfully")

        self.data.unload_data()
        self.data.serialize(OUTPUT_DATA_PATH_DOCKER)
    