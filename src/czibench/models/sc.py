from abc import ABC, abstractmethod
from typing import ClassVar, List
from .base import BaseModel
from ..datasets.sc import SingleCellDataset
from ..datasets.types import Organism
import logging

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
                f"Can't instantiate {cls.__name__} without available_organisms class variable"
            )

        if not hasattr(cls, "required_obs_keys"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} without required_obs_keys class variable"
            )
        
        if not hasattr(cls, "required_var_keys"):
            raise TypeError(
                f"Can't instantiate {cls.__name__} without required_var_keys class variable"
            )

    @classmethod
    @abstractmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset):
        pass

    @classmethod
    def _validate_dataset(cls, dataset: SingleCellDataset):
        if dataset.organism not in cls.available_organisms:
            raise ValueError(f"Dataset organism {dataset.organism} is not supported for {cls.__name__}")

        cls._validate_model_requirements(dataset)

class ScviValidator(BaseSingleCell, ABC):
    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]
    required_var_keys = []

    @classmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset):
        # Check if all required batch keys are present in obs
        missing_keys = [
            key for key in cls.required_obs_keys if key not in dataset.adata.obs.columns
        ]

        if missing_keys:
            raise ValueError(f"Missing required obs keys: {missing_keys}")
        
        missing_keys = [
            key for key in cls.required_var_keys if key not in dataset.adata.var.columns
        ]

        if missing_keys:
            raise ValueError(f"Missing required var keys: {missing_keys}")

        return True

class UCEValidator(BaseSingleCell, ABC):
    available_organisms = [Organism.HUMAN, Organism.MOUSE] # TODO: add other UCE organisms
    required_obs_keys = []
    required_var_keys = ["gene_symbol"]

    @classmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset):
        missing_keys = [
            key for key in cls.required_var_keys if key not in dataset.adata.var.columns
        ]

        if missing_keys:
            raise ValueError(f"Missing required var keys: {missing_keys}")

        return True
class ScGPTValidator(BaseSingleCell, ABC):
    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["gene_symbol"]

    @classmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset):
        # Check if all required var keys are present in var
        missing_keys = [
            key for key in cls.required_var_keys if key not in dataset.adata.var.columns
        ]

        if missing_keys:
            raise ValueError(f"Missing required var keys: {missing_keys}")
