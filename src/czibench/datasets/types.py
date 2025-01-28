from enum import Enum
from omegaconf import OmegaConf


class Organism(Enum):
    HUMAN = ("homo_sapiens", "ENSG")
    MOUSE = ("mus_musculus", "ENSMUSG")

    # Todo: add other organisms
    def __init__(self, name: str, prefix: str):
        self._value_ = (name, prefix)  # This is handled automatically by Enum

    def __str__(self):
        return self.value[0]  # Access the name from the tuple

    def __repr__(self):
        return self.value[0]  # Access the name from the tuple

    @property
    def name(self):
        return self.value[0]

    @property
    def prefix(self):
        return self.value[1]


# Register Organism resolver
OmegaConf.register_new_resolver("organism", lambda name: getattr(Organism, name))
