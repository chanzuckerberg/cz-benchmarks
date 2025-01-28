import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Type

from ..constants import INPUT_DATA_PATH_DOCKER, OUTPUT_DATA_PATH_DOCKER
from ..datasets.base import BaseDataset

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # This ensures the configuration is applied
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseModel(ABC):
    # Type annotation for class variable
    dataset_type: ClassVar[Type[BaseDataset]]
    data: BaseDataset

    def __init_subclass__(cls) -> None:
        """Validate that subclasses define required class variables"""
        super().__init_subclass__()
        if not hasattr(cls, "dataset_type"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} without dataset_type class"
                " variable"
            )

    @abstractmethod
    def _validate_dataset(self, dataset: BaseDataset) -> bool:
        pass

    @classmethod
    def validate_dataset(cls, dataset: BaseDataset):
        if not isinstance(dataset, cls.dataset_type):
            raise ValueError(
                "Dataset type mismatch: expected"
                f" {cls.dataset_type.__name__}, got {type(dataset).__name__}"
            )

        cls._validate_dataset(dataset)

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

        logger.info("Running model...")
        self.run_model()
        logger.info("Model ran successfully")

        self.data.unload_data()
        self.data.serialize(OUTPUT_DATA_PATH_DOCKER)
