from typing import Set

from ...datasets.types import DataType, Organism
from .base_single_cell_model_validator import BaseSingleCellValidator


class ScGenePTValidator(BaseSingleCellValidator):
    """Validation requirements for ScGenePT models.

    Validates datasets for use with Single-cell Gene Perturbation Transformer models.
    Requires gene symbols and currently only supports human data.
    Used for perturbation prediction tasks.

    Class Variables:
        available_organisms: Only human data supported
        required_obs_keys: No observation requirements
        required_var_keys: Must have gene_symbol column
    """

    available_organisms = [Organism.HUMAN]
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
            Set containing perturbation predictions
        """
        return {DataType.PERTURBATION}
