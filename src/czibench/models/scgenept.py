from ..datasets.types import Organism
from .single_cell import BaseSingleCellValidator


class ScGenePTValidator(BaseSingleCellValidator):
    """Validation requirements for ScGenePT models."""

    available_organisms = [Organism.HUMAN]
    required_obs_keys = []
    required_var_keys = ["gene_symbol"]
