from typing import Set

from ...datasets.types import DataType, Organism
from .base_single_cell_model_validator import BaseSingleCellValidator
from ..types import ModelType


class GeneformerValidator(BaseSingleCellValidator):
    """Validation requirements for Geneformer models.

    Validates datasets for use with Geneformer transformer models.
    Requires feature IDs and currently only supports human data.
    """

    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["feature_id"]
    model_type = ModelType.register("GENEFORMER")

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
