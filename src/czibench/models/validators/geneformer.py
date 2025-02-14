from ...datasets.types import Organism
from ..single_cell import BaseSingleCellValidator


class GeneformerValidator(BaseSingleCellValidator):
    """Validation requirements for Geneformer models."""

    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["feature_id"]
