# import os
from abc import ABC, abstractmethod
from typing import Any#, Optional, Dict, Type
# import dill


from .types import Organism # DataType, DataValue, 
# from ..models.types import ModelType, ModelOutputs


class BaseDataset(ABC):
    def __init__(self, path: str, organism: Organism, **kwargs: Any):
        # self._inputs: Dict[DataType, DataValue] = {}
        # self._outputs: ModelOutputs = {}

        self.path = path
        # self.set_input(DataType.ORGANISM, organism)
        # FIXME VALIDATION: validate that organism name(s) are in dataset validator
        self.organism = organism
        self.kwargs = kwargs

        for key, value in kwargs.items():
            setattr(self, key, value)

    # @property
    # def inputs(self) -> Dict[DataType, DataValue]:
    #     """Get the inputs dictionary."""
    #     return self._inputs

    # @property
    # def outputs(self) -> ModelOutputs:
    #     """Get the outputs dictionary."""
    #     return self._outputs

    # @property
    # def organism(self) -> Organism:
    #     return self.get_input(DataType.ORGANISM)

    # def _validate_type(
    #     self, value: Any, expected_type: Type, type_name: str = ""
    # ) -> None:
    #     """Helper function to validate types including generics.

    #     Args:
    #         value: The value to check
    #         expected_type: The expected type (can be generic like Dict[str, DataFrame])
    #         type_name: Optional name of the type for error messages
    #     """
    #     # Check if it's a generic type
    #     if hasattr(expected_type, "__origin__"):
    #         base_type = expected_type.__origin__
    #         type_args = expected_type.__args__

    #         # Special handling for Dict types
    #         if base_type is dict:
    #             if not isinstance(value, dict):
    #                 raise TypeError(
    #                     f"{type_name} has incorrect type: "
    #                     f"expected {expected_type}, got {type(value)}"
    #                 )
    #             # Check key and value types
    #             key_type, value_type = type_args
    #             for k, v in value.items():
    #                 if not isinstance(k, key_type):
    #                     raise TypeError(
    #                         "Dict key has incorrect type:"
    #                         f"expected {key_type}, got {type(k)}"
    #                     )
    #                 if not isinstance(v, value_type):
    #                     raise TypeError(
    #                         "Dict value has incorrect type:"
    #                         f"expected {value_type}, got {type(v)}"
    #                     )
    #         else:
    #             # Handle other generic types if needed
    #             if not isinstance(value, base_type):
    #                 raise TypeError(
    #                     f"{type_name} has incorrect type: "
    #                     f"expected {expected_type}, got {type(value)}"
    #                 )
    #     else:
    #         # Non-generic types
    #         if not isinstance(value, expected_type):
    #             raise TypeError(
    #                 f"{type_name} has incorrect type: "
    #                 f"expected {expected_type}, got {type(value)}"
    #             )

    # def set_input(self, data_type: DataType, value: DataValue) -> None:
    #     """Safely set an input with type checking."""
    #     if not data_type.is_input:
    #         raise ValueError(f"Cannot set output type as input: {data_type.name}")
    #     self._validate_type(value, data_type.dtype, f"Input {data_type.name}")
    #     self._inputs[data_type] = value

    # def set_output(
    #     self, model_type: ModelType | None, data_type: DataType, value: DataValue
    # ) -> None:
    #     """Safely set an output with type checking.
    #     Args:
    #         model_type (ModelType | None): The type of model associated with the output.
    #             This parameter is used to differentiate between outputs
    #             from various models. It can be set to `None` if the output
    #             is not tied to a specific model type defined in the `ModelType` enum.
    #         data_type (DataType): Specifies the data type of the output.
    #         value (Any): The value to assign to the output.
    #     """
    #     if data_type.is_input:
    #         raise ValueError(f"Cannot set input type as output: {data_type.name}")

    #     self._validate_type(value, data_type.dtype, f"Output {data_type.name}")
    #     if model_type not in self._outputs:
    #         self._outputs[model_type] = {}
    #     self._outputs[model_type][data_type] = value

    # def get_input(self, data_type: DataType) -> DataValue:
    #     """Safely get an input with error handling."""
    #     if data_type not in self._inputs:
    #         raise KeyError(f"Input {data_type.name} not found")
    #     return self._inputs[data_type]

    # def get_output(
    #     self, model_type: ModelType | None, data_type: DataType
    # ) -> DataValue:
    #     """Safely get an output with error handling.
    #     Args:
    #         model_type (ModelType | None): The type of model associated with the output.
    #             This parameter is used to differentiate between outputs
    #             from various models. It can be set to `None` if the output
    #             is not tied to a specific model type defined in the `ModelType` enum.
    #         data_type (DataType): Specifies the data type of the output.
    #     Returns:
    #         DataValue: The value of the output.
    #     """
    #     if model_type not in self._outputs:
    #         raise KeyError(f"Outputs for model {model_type.name} not found")
    #     if data_type not in self._outputs[model_type]:
    #         raise KeyError(
    #             f"Output {data_type.name} not found for model {model_type.name}"
    #         )
    #     return self._outputs[model_type][data_type]

    # @abstractmethod
    # def _validate(self) -> None:
    #     pass

    # def validate(self) -> None:
    #     """Validate that all inputs and outputs match their expected types"""

    #     if not os.path.exists(self.path):
    #         raise ValueError("Dataset path does not exist")

    #     for data_type, value in self.inputs.items():
    #         self._validate_type(value, data_type.dtype, f"Input {data_type.name}")

    #     for model_outputs in self.outputs.values():
    #         for data_type, value in model_outputs.items():
    #             self._validate_type(value, data_type.dtype, f"Output {data_type.name}")

    #     self._validate()

    @abstractmethod
    def load_data(self) -> None:
        """
        Load the dataset into memory.

        This method should be implemented by subclasses to load their specific
        data format.
        For example, SingleCellDataset loads an AnnData object from an h5ad
        file.

        The loaded data should be stored as instance attributes that can be
        accessed by other methods.
        """

    @abstractmethod
    def unload_data(self) -> None:
        """
        Unload the dataset from memory.

        This method should be implemented by subclasses to free memory by
        clearing loaded data.
        For example, SingleCellDataset sets its AnnData object to None.

        This is used to clear memory-intensive data before serialization,
        since serializing large raw data artifacts can be error-prone and
        inefficient.

        Any instance attributes containing loaded data should be cleared or
        set to None.
        """

    # def serialize(self, path: str) -> None:
    #     """
    #     Serialize this dataset instance to disk using dill.

    #     Args:
    #         path: Path where the serialized dataset should be saved
    #     """
    #     if not path.endswith(".dill"):
    #         path = f"{path}.dill"

    #     with open(path, "wb") as f:
    #         dill.dump(self, f)

    # @staticmethod
    # def deserialize(path: str) -> "BaseDataset":
    #     """
    #     Load a serialized dataset from disk.

    #     Args:
    #         path: Path to the serialized dataset file

    #     Returns:
    #         BaseDataset: The deserialized dataset instance
    #     """
    #     if not path.endswith(".dill"):
    #         path = f"{path}.dill"

    #     if not os.path.exists(path):
    #         raise FileNotFoundError(f"Dataset file not found at {path}")

    #     with open(path, "rb") as f:
    #         return dill.load(f)
