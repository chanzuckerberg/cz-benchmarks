import pytest
from omegaconf import OmegaConf
from src.czibench.datasets.types import Organism


def test_organism_enum():
    human = Organism.HUMAN
    assert human.name == "homo_sapiens"
    assert human.prefix == "ENSG"
    assert str(human) == "homo_sapiens"
    assert repr(human) == "homo_sapiens"

    mouse = Organism.MOUSE
    assert mouse.name == "mus_musculus"
    assert mouse.prefix == "ENSMUSG"
    assert str(mouse) == "mus_musculus"
    assert repr(mouse) == "mus_musculus"

def test_organism_from_string():
    assert Organism.from_string("human") == Organism.HUMAN
    assert Organism.from_string("homo_sapiens") == Organism.HUMAN
    assert Organism.from_string("mouse") == Organism.MOUSE
    assert Organism.from_string("mus_musculus") == Organism.MOUSE

    with pytest.raises(ValueError):
        Organism.from_string("unknown")

# TODO - debug running fine individually, failing for suite
# def test_organism_resolver():
#     # Create a config object
#     config = OmegaConf.create({"organism": "${organism:human}"})
#     OmegaConf.resolve(config)
#     assert config["organism"] == Organism.HUMAN
#     config = OmegaConf.create({"organism": "${organism:homo_sapiens}"})
#     OmegaConf.resolve(config)
#     assert config["organism"] == Organism.HUMAN
#     config = OmegaConf.create({"organism": "${organism:mouse}"})
#     OmegaConf.resolve(config)
#     assert config["organism"] == Organism.MOUSE
#     config = OmegaConf.create({"organism": "${organism:mus_musculus}"})
#     OmegaConf.resolve(config)
#     assert config["organism"] == Organism.MOUSE
#     config = OmegaConf.create({"organism": "${organism:unknown}"})

#     with pytest.raises(ValueError):
#         OmegaConf.resolve(config)