import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Set, Type

from ...datasets import BaseDataset
from ..types import ModelType

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,  # This ensures the configuration is applied
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BaseModelValidator(ABC):
    """Abstract base class for model validators.

    Defines the interface for validating datasets against model requirements.
    Validators ensure datasets meet model-specific requirements like:
    - Compatible data types
    - Required metadata fields
    - Organism compatibility
    - Feature name formats

    Each validator must:
    1. Define a dataset_type class variable
    2. Define a model_type class variable or model_name property
    3. Implement _validate_dataset, inputs, and outputs
    """

    # Type annotation for class variables
    dataset_type: ClassVar[Type[BaseDataset]]
    model_type: ClassVar[ModelType]

    def __init_subclass__(cls) -> None:
        """Validate that subclasses define required class variables and
        follow naming conventions.

        Raises:
            TypeError: If required class variables are missing or invalid
            ValueError: If class naming doesn't follow conventions
        """
        super().__init_subclass__()

        if cls.__name__ == "BaseModelImplementation":
            return

        # Check for dataset_type
        if not hasattr(cls, "dataset_type"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} without dataset_type class variable"
            )

    @abstractmethod
    def _validate_dataset(self, dataset: BaseDataset):
        """Perform model-specific dataset validation.

        Args:
            dataset: Dataset to validate

        Raises:
            ValueError: If validation fails
        """

    @property
    @abstractmethod
    def inputs(self):
        """Required input data types this model requires.

        Returns:
            Set of required DataType enums
        """

    @property
    @abstractmethod
    def outputs(self):
        """Output data types produced by this model.

        Returns:
            Set of output DataType enums
        """

    def validate_dataset(self, dataset: BaseDataset):
        """Validate a dataset meets all model requirements.

        Checks:
        1. Dataset type matches model requirements
        2. Required inputs are available
        3. Model-specific validation rules

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

        # Validate required inputs are available
        missing_inputs = self.inputs - set(dataset.inputs.keys())
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")

        self._validate_dataset(dataset)
