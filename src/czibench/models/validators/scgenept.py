from typing import Set
from ...datasets.types import Organism, DataType
from ..single_cell import BaseSingleCellValidator


class ScGenePTValidator(BaseSingleCellValidator):
    """Validation requirements for ScGenePT models."""

    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["gene_symbol"]

    @property
    def inputs(self) -> Set[DataType]:
        return {DataType.ANNDATA}

    @property
    def outputs(self) -> Set[DataType]:
        return {
            DataType.PERTURBATION_PRED,
            DataType.PERTURBATION_TRUTH,
            DataType.PERTURBATION_CONTROL,
        }
