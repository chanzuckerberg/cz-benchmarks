from typing import Set

from ...datasets.types import DataType, Organism
from .base_single_cell_model_validator import BaseSingleCellValidator
from ..types import ModelType


class SCVIValidator(BaseSingleCellValidator):
    """Validation requirements for scVI models.

    Validates datasets for use with Single-cell Variational Inference models.
    Requires detailed metadata about the dataset, assay, and donor information.
    Supports both human and mouse data.

    """

    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]
    required_var_keys = []
    model_type = ModelType.SCVI

    @property
    def inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set containing AnnData and metadata requirements
        """
        return {DataType.ANNDATA, DataType.METADATA}

    @property
    def outputs(self) -> Set[DataType]:
        """Expected model output types.

        Returns:
            Set containing embedding output type
        """
        return {DataType.EMBEDDING}
