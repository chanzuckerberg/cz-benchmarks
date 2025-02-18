from typing import Set
from ...datasets.types import Organism
from ..single_cell import BaseSingleCellValidator
from ...datasets.types import DataType


class SCVIValidator(BaseSingleCellValidator):
    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]
    required_var_keys = []

    @property
    def inputs(self) -> Set[DataType]:
        return {DataType.ANNDATA, DataType.METADATA}

    @property
    def outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}
