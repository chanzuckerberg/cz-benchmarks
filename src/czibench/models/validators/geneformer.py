from typing import Set
from ...datasets.types import Organism, DataType
from ..single_cell import BaseSingleCellValidator


class GeneformerValidator(BaseSingleCellValidator):
    """Validation requirements for Geneformer models."""

    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["feature_id"]

    @property
    def inputs(self) -> Set[DataType]:
        return {DataType.ANNDATA}

    @property
    def outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}
