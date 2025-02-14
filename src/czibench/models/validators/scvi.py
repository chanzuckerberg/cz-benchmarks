from ...datasets.types import Organism
from ..single_cell import BaseSingleCellValidator


class SCVIValidator(BaseSingleCellValidator):
    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]
    required_var_keys = []
