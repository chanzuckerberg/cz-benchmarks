from ..datasets.types import Organism
from .single_cell import BaseSingleCellValidator


class UCEValidator(BaseSingleCellValidator):
    """Validation requirements for UCE models."""

    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = []
    required_var_keys = ["gene_symbol"]
