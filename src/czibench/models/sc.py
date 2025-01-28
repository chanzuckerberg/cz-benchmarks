import logging
from abc import ABC, abstractmethod
from typing import ClassVar, List

from ..datasets.sc import SingleCellDataset
from ..datasets.types import Organism
from .base import BaseModel

logger = logging.getLogger(__name__)


class BaseSingleCell(BaseModel, ABC):
    dataset_type = SingleCellDataset
    available_organisms: ClassVar[List[Organism]]
    required_obs_keys: ClassVar[List[str]]
    required_var_keys: ClassVar[List[str]]

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not hasattr(cls, "available_organisms"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} without"
                " available_organisms class variable"
            )

        if not hasattr(cls, "required_obs_keys"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} without required_obs_keys"
                " class variable"
            )

        if not hasattr(cls, "required_var_keys"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} without required_var_keys"
                " class variable"
            )

    @classmethod
    @abstractmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset):
        pass

    @classmethod
    def _validate_dataset(cls, dataset: SingleCellDataset):
        if dataset.organism not in cls.available_organisms:
            raise ValueError(
                f"Dataset organism {dataset.organism} is not supported for"
                f" {cls.__name__}"
            )

        cls._validate_model_requirements(dataset)

