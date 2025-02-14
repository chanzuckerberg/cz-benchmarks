from typing import ClassVar, List
from .base import BaseModelValidator
from ..datasets.sc import SingleCellDataset
from ..datasets.types import Organism


class BaseSingleCellValidator(BaseModelValidator):
    dataset_type = SingleCellDataset
    available_organisms: ClassVar[List[Organism]]
    required_obs_keys: ClassVar[List[str]]
    required_var_keys: ClassVar[List[str]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not hasattr(cls, "available_organisms"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} "
                "without available_organisms class variable"
            )

        if not hasattr(cls, "required_obs_keys"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} "
                "without required_obs_keys class variable"
            )

        if not hasattr(cls, "required_var_keys"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} "
                "without required_var_keys class variable"
            )

    def _validate_dataset(self, dataset: SingleCellDataset):
        if dataset.organism not in self.available_organisms:
            raise ValueError(
                f"Dataset organism {dataset.organism} "
                "is not supported for {self.__class__.__name__}"
            )

        missing_keys = [
            key
            for key in self.required_obs_keys
            if key not in dataset.adata.obs.columns
        ]

        if missing_keys:
            raise ValueError(f"Missing required obs keys: {missing_keys}")

        missing_keys = [
            key
            for key in self.required_var_keys
            if key not in dataset.adata.var.columns
        ]

        if missing_keys:
            raise ValueError(f"Missing required var keys: {missing_keys}")
