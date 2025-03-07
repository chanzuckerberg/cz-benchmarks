from enum import Enum
from typing import Dict

from ..datasets.types import DataType, DataValue


class ModelType:
    """Registry for model types."""
    
    _registry = {}

    def __init__(self, name: str):
        self.name = name

    @classmethod
    def register(cls, name: str) -> "ModelType":
        """Register a new model type."""
        if name in cls._registry:
            return cls._registry[name]
        model_type = cls(name)
        cls._registry[name] = model_type
        return model_type

    @classmethod
    def __getattr__(cls, name: str) -> "ModelType":
        """Allow access to registered types via dot notation."""
        if name.isupper():
            if name in cls._registry:
                return cls._registry[name]
            raise AttributeError(f"Model type '{name}' is not registered")
        raise AttributeError(f"'{cls.__name__}' has no attribute '{name}'")

    def __eq__(self, other):
        if isinstance(other, ModelType):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


# Pre-register baseline
ModelType.BASELINE = ModelType.register("BASELINE")

# Type alias for model outputs
ModelOutputs = Dict[ModelType, Dict[DataType, DataValue]]
