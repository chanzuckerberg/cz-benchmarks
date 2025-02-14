from ...datasets.types import Organism
from ..single_cell import BaseSingleCellValidator


class ScGPTValidator(BaseSingleCellValidator):
    """Validation requirements for ScGPT models."""

    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["gene_symbol"]
