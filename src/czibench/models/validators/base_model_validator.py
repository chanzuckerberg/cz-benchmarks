import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Set, Type

from ...datasets.base import BaseDataset
from ...datasets.types import DataType
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

        # Validate class name follows convention
        if not cls.__name__.endswith("Validator"):
            raise ValueError(
                f"Validator class {cls.__name__} must end with 'Validator'"
            )

        # Check for dataset_type
        if not hasattr(cls, "dataset_type"):
            raise TypeError(
                f"Can't instantiate {cls.__name__}"
                " without dataset_type class variable"
            )

        # Handle model type registration
        if hasattr(cls, "model_type"):
            # If model_type is explicitly defined, use it
            if not isinstance(cls.model_type, ModelType):
                raise TypeError(
                    f"model_type in {cls.__name__} must be a ModelType enum value"
                )
        elif hasattr(cls, "model_name"):
            # If model_name property is defined, use it for registration
            model_name = cls.model_name
            if not isinstance(model_name, str):
                raise TypeError(f"model_name in {cls.__name__} must be a string")
            cls.model_type = ModelType.register(model_name.upper())
        else:
            # Try to derive from class name as fallback
            model_name = cls.__name__.replace("Validator", "")
            if not model_name:
                raise ValueError(
                    f"Could not derive model name from class {cls.__name__}"
                )
            cls.model_type = ModelType.register(model_name.upper())

    @property
    def model_name(self) -> str:
        """Get the model name for registration.

        This property can be overridden by subclasses to provide a custom model name.
        By default, it derives the name from the class name.

        Returns:
            The model name to use for registration
        """
        return self.__class__.__name__.replace("Validator", "")

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
    def inputs(self) -> Set[DataType]:
        """Required input data types this model requires.

        Returns:
            Set of required DataType enums
        """

    @property
    @abstractmethod
    def outputs(self) -> Set[DataType]:
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
        if not isinstance(dataset, self.dataset_type):
            raise ValueError("Dataset type mismatch")

        # Validate required inputs are available
        missing_inputs = self.inputs - set(dataset.inputs.keys())
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")

        self._validate_dataset(dataset)
