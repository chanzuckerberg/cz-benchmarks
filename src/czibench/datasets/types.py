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
    
    @staticmethod
    def from_string(s):
        s = s.lower()
        if s in ["human", "homo_sapiens"]:
            return Organism.HUMAN
        elif s in ["mouse", "mus_musculus"]:
            return Organism.MOUSE
        else:
            raise ValueError(f"Unknown organism: {s}")


# Register Organism resolver
print("Before Organism resolver registration. Is resolver registered? : ", OmegaConf.has_resolver('organism'))
if not OmegaConf.has_resolver('organism'): 
    print("Doing Organism resolver registration")
    OmegaConf.register_new_resolver("organism", lambda name: Organism.from_string(name))
