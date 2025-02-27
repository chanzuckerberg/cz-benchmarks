from typing import Set

from ...datasets.types import DataType, Organism
from .base_single_cell_model_validator import BaseSingleCellValidator


class GeneformerValidator(BaseSingleCellValidator):
    """Validation requirements for Geneformer models.

    Validates datasets for use with Geneformer transformer models.
    Requires feature IDs and currently only supports human data.

    Class Variables:
        available_organisms: Only human data supported
        required_obs_keys: No observation requirements
        required_var_keys: Must have feature_id column for gene identifiers
    """

    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["feature_id"]

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
