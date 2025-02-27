from typing import Set

from ...datasets.types import DataType, Organism
from .base_single_cell_model_validator import BaseSingleCellValidator


class UCEValidator(BaseSingleCellValidator):
    """Validation requirements for UCE models.

    Validates datasets for use with Universal Cell Embeddings (UCE) models.
    Requires gene symbols and supports both human and mouse data.

    Class Variables:
        available_organisms: Supported species (human and mouse)
        required_obs_keys: No observation requirements
        required_var_keys: Must have gene_symbol column
    """

    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = []
    required_var_keys = ["gene_symbol"]

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
