from enum import Enum
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from ..datasets.types import DataType, DataValue
else:
    DataType = "DataType"
    DataValue = "DataValue"


class ModelType(Enum):
    BASELINE = "BASELINE"
    SCVI = "SCVI"
    SCGPT = "SCGPT"
    GENEFORMER = "GENEFORMER"
    SCGENEPT = "SCGENEPT"
    UCE = "UCE"
    AIDO = "AIDO"
    TRANSCRIPTFORMER = "TRANSCRIPTFORMER"
    SIMPLE = "SIMPLE"

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return super().__eq__(other)

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.__class__.__name__}.{self.value}"


# Type alias for model outputs
ModelOutputs = Dict[ModelType, Dict[DataType, DataValue]]
