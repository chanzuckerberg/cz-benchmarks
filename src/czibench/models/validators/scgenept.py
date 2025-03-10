from typing import Set

from ...datasets import DataType, Organism
from .base_single_cell_model_validator import BaseSingleCellValidator
from ..types import ModelType


class ScGenePTValidator(BaseSingleCellValidator):
    """Validation requirements for ScGenePT models.

    Validates datasets for use with Single-cell Gene Perturbation Transformer models.
    Requires gene symbols and currently only supports human data.
    Used for perturbation prediction tasks.
    """

    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["feature_name"]
    model_type = ModelType.SCGENEPT

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
            Set containing perturbation predictions and ground truth values for
            evaluating perturbation prediction performance
        """
        return {
            DataType.PERTURBATION_PRED,
            DataType.PERTURBATION_TRUTH,
        }
