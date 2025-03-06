from enum import Enum
from typing import Dict

from ..datasets.types import DataType, DataValue


class ModelType(Enum):
    """Registry for model types.

    This enum serves as a registry for all model types in the system.
    Each model validator should register itself by adding a new enum value.
    """

    BASELINE = "baseline"

    @classmethod
    def register(cls, name: str) -> "ModelType":
        """Register a new model type.

        Args:
            name: Name of the model type to register

        Returns:
            The newly created ModelType enum value
        """
        if name in cls.__members__:
            return cls[name]
        return cls._value2member_map_.setdefault(
            len(cls._value2member_map_) + 1,
            cls._member_class_(
                cls,
                name,
                len(cls._value2member_map_) + 1,
            ),
        )


# Type alias for model outputs
ModelOutputs = Dict[ModelType, Dict[DataType, DataValue]]
