from typing import ClassVar, List

from ...datasets import SingleCellDataset, Organism
from .base_model_validator import BaseModelValidator


class BaseSingleCellValidator(BaseModelValidator):
    """Base validator for single-cell models.

    Defines common validation logic for single-cell models, including:
    - Organism compatibility checking
    - Required metadata validation
    - AnnData observation and variable key validation
    """

    dataset_type = SingleCellDataset
    available_organisms: ClassVar[List[Organism]]
    required_obs_keys: ClassVar[List[str]]
    required_var_keys: ClassVar[List[str]]

    def __init_subclass__(cls) -> None:
        """Validate required class variables in child classes.

        Ensures child classes define:
        - available_organisms
        - required_obs_keys
        - required_var_keys

        Raises:
            TypeError: If any required class variable is missing
        """
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
        """Validate a single-cell dataset.

        Checks:
        1. Dataset organism is supported
        2. Required observation keys are present
        3. Required variable keys are present

        Args:
            dataset: SingleCellDataset to validate

        Raises:
            ValueError: If validation fails
        """
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
