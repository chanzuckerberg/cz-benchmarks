from typing import Set
from ...datasets.types import Organism, DataType
from ..single_cell import BaseSingleCellValidator


class UCEValidator(BaseSingleCellValidator):
    """Validation requirements for UCE models."""

    available_organisms = [
        Organism.HUMAN,  # Homo sapiens
        Organism.MOUSE,  # Mus musculus
        Organism.TROPICAL_CLAWED_FROG,  # Xenopus tropicalis
        Organism.ZEBRAFISH,  # Danio rerio
        Organism.MOUSE_LEMUR,  # Microcebus murinus
        Organism.WILD_BOAR,  # Sus scrofa
        Organism.CRAB_EATING_MACAQUE,  # Macaca fascicularis
        Organism.RHESUS_MACAQUE,  # Macaca mulatta
    ]
    required_obs_keys = []
    required_var_keys = ["gene_symbol"]

    @property
    def inputs(self) -> Set[DataType]:
        return {DataType.ANNDATA}

    @property
    def outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}
