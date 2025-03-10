from typing import Set

from ...datasets import DataType, Organism
from .base_single_cell_model_validator import BaseSingleCellValidator
from ..types import ModelType


class ScGPTValidator(BaseSingleCellValidator):
    """Validation requirements for ScGPT models.

    Validates datasets for use with Single-cell GPT models.
    Requires gene symbols and currently only supports human data.

    """

    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["feature_name"]
    model_type = ModelType.SCGPT

    @property
    def inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set containing AnnData requirement
        """
        return {DataType.ANNDATA}

    @property
    def outputs(self) -> Set[DataType]:
        """Expected model output types.

        Returns:
            Set containing embedding output type
        """
        return {DataType.EMBEDDING}
