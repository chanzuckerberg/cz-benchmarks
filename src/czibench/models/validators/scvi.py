from typing import Set

from ...datasets.types import DataType, Organism
from ..single_cell import BaseSingleCellValidator


class SCVIValidator(BaseSingleCellValidator):
    """Validation requirements for scVI models.

    Validates datasets for use with Single-cell Variational Inference models.
    Requires detailed metadata about the dataset, assay, and donor information.
    Supports both human and mouse data.

    Class Variables:
        available_organisms: Supports human and mouse data
        required_obs_keys: Required metadata columns:
            - dataset_id: Identifier for the dataset
            - assay: Type of assay used
            - suspension_type: Sample preparation method
            - donor_id: Sample donor identifier
        required_var_keys: No variable requirements
    """

    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]
    required_var_keys = []

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
