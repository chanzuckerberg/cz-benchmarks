from enum import Enum

from omegaconf import OmegaConf


class Organism(Enum):
    HUMAN = ("homo_sapiens", "ENSG")
    MOUSE = ("mus_musculus", "ENSMUSG")
    TROPICAL_CLAWED_FROG = ("xenopus_tropicalis", "ENSXETG")
    AFRICAN_CLAWED_FROG = ("xenopus_laevis", "ENSXLAG")
    ZEBRAFISH = ("danio_rerio", "ENSDARG")
    MOUSE_LEMUR = ("microcebus_murinus", "ENSMICG")
    WILD_BOAR = ("sus_scrofa", "ENSSSCG")
    CRAB_EATING_MACAQUE = ("macaca_fascicularis", "ENSMFAG")
    RHESUS_MACAQUE = ("macaca_mulatta", "ENSMMUG")
    PLATYPUS = ("ornithorhynchus_anatinus", "ENSOANG")
    OPOSSUM = ("monodelphis_domestica", "ENSMODG")
    GORILLA = ("gorilla_gorilla", "ENSGGOG")
    CHIMPANZEE = ("pan_troglodytes", "ENSPTRG")
    MARMOSET = ("callithrix_jacchus", "ENSCJAG")
    CHICKEN = ("gallus_gallus", "ENSGALG")
    RABBIT = ("oryctolagus_cuniculus", "ENSOCUG")
    FRUIT_FLY = ("drosophila_melanogaster", "FBgn")
    RAT = ("rattus_norvegicus", "ENSRNOG")
    NAKED_MOLE_RAT = ("heterocephalus_glaber", "ENSHGLG")
    CAENORHABDITIS_ELEGANS = ("caenorhabditis_elegans", "WBGene")
    YEAST = ("saccharomyces_cerevisiae", "")
    MALARIA_PARASITE = ("plasmodium_falciparum", "PF3D7")
    SEA_LAMPREY = ("petromyzon_marinus", "ENSPMAG")
    FRESHWATER_SPONGE = ("spongilla_lacustris", "ENSLPGG")
    CORAL = ("stylophora_pistillata", "LOC")
    SEA_URCHIN = ("lytechinus_variegatus", "")  # Mixed prefixes: LOC and GeneID

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
if not OmegaConf.has_resolver("organism"):  # Required for dataset test cases
    OmegaConf.register_new_resolver("organism", lambda name: getattr(Organism, name))
