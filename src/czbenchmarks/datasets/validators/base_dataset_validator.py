import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Type

from ...datasets import Dataset


# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # This ensures the configuration is applied
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseDatasetValidator(ABC):
    """Abstract base class for dataset validators. Not used in code and provided as convenience to validate user datasets.

    Defines the interface for validating datasets against dataset requirements.
    Validators ensure datasets meet dataset-specific requirements like:
    - Compatible data types
    - Organism compatibility
    - Feature name formats

    Each validator must:
    1. Define a dataset_type class variable
    2. Implement _validate_dataset, inputs, and outputs as abstract methods/properties
    """

    # Type annotation for class variables
    dataset_type: ClassVar[Type[Dataset]]

    def __init_subclass__(cls) -> None:
        """Validate that subclasses define required class variables.

        Raises:
            TypeError: If required class variables are missing
        """
        super().__init_subclass__()

        # Check for dataset_type
        if not hasattr(cls, "dataset_type"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} without dataset_type class variable"
            )

    @abstractmethod
    def _validate_dataset(self, dataset: Dataset):
        """Perform dataset-specific validation.

        Args:
            dataset: Dataset to validate

        Raises:
            ValueError: If validation fails
        """

    def validate_dataset(self, dataset: Dataset):
        """Validate that a dataset meets all requirements.

        Checks:
        1. Dataset type matches dataset_type
        2. Runs dataset specific validation

        Args:
            dataset: Dataset to validate

        Raises:
            ValueError: If validation fails
        """
        if type(dataset) is not self.dataset_type:
            raise ValueError(
                f"Dataset type mismatch. Expected {self.dataset_type.__name__}, "
                f"got {type(dataset).__name__}"
            )

        self._validate_dataset(dataset)
