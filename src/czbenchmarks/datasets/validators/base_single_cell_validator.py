from typing import ClassVar, List

from ...datasets import SingleCellDataset, Organism
from .base_dataset_validator import BaseDatasetValidator


class BaseSingleCellValidator(BaseDatasetValidator):
    """Base validator for single-cell datasets.

    Provides validation logic for single-cell datasets, including:
    - Checking if the dataset organism is supported
    - Validating presence of required observation and variable keys in AnnData
    """

    dataset_type = SingleCellDataset
    available_organisms: ClassVar[List[Organism]]
    required_obs_keys: ClassVar[List[str]]
    required_var_keys: ClassVar[List[str]]

    def __init_subclass__(cls) -> None:
        """Ensure required class variables are defined in subclasses.

        Subclasses must define:
        - available_organisms
        - required_obs_keys
        - required_var_keys

        Raises:
            TypeError: If any required class variable is missing in the subclass
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
        1. Dataset organism is in available_organisms
        2. All required observation keys are present in dataset.adata.obs.columns
        3. All required variable keys are present in dataset.adata.var.columns

        Args:
            dataset: SingleCellDataset to validate

        Raises:
            ValueError: If any validation check fails
        """
        if dataset.organism not in self.available_organisms:
            raise ValueError(
                f"Dataset organism {dataset.organism} "
                "is not supported for {self.__class__.__name__}"
            )

        # Check for missing required observation keys in AnnData obs
        missing_keys = [
            key
            for key in self.required_obs_keys
            if key not in dataset.adata.obs.columns
        ]

        if missing_keys:
            raise ValueError(f"Missing required obs keys: {missing_keys}")

        # Check for missing required variable keys in AnnData var
        missing_keys = [
            key
            for key in self.required_var_keys
            if key not in dataset.adata.var.columns
        ]

        if missing_keys:
            raise ValueError(f"Missing required var keys: {missing_keys}")
