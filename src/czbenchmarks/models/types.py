from enum import Enum
from typing import Dict
from ..datasets.types import DataType, DataValue


class ModelType(Enum):
    BASELINE = "BASELINE"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return super().__eq__(other)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


# Type alias for model outputs
ModelOutputs = Dict[ModelType, Dict[DataType, DataValue]]
