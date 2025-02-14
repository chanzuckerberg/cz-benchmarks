from .single_cell import BaseSingleCellValidator
from ..datasets.sc import SingleCellDataset
from ..datasets.types import Organism

class SCVIValidator(BaseSingleCellValidator):
    available_organisms = [Organism.HUMAN, Organism.MOUSE]
    required_obs_keys = ["dataset_id", "assay", "suspension_type", "donor_id"]
    required_var_keys = []