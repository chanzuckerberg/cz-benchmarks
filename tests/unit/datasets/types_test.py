import anndata as ad
import numpy as np
import pandas as pd
import pytest
from czibench.datasets.types import DataType, Organism
from omegaconf import OmegaConf


@pytest.fixture
def example_anndata():
    return ad.read_h5ad("tests/assets/example-small.h5ad")


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


def test_omegaconf_organism_resolver():
    cfg = OmegaConf.create({"org": "${organism:HUMAN}"})
    assert cfg.org == Organism.HUMAN
    cfg = OmegaConf.create({"org": "${organism:MOUSE}"})
    assert cfg.org == Organism.MOUSE


def test_datatype_members():
    assert DataType.METADATA.dtype == pd.DataFrame
    assert DataType.ANNDATA.dtype == ad.AnnData
    assert DataType.ORGANISM.dtype == Organism
    assert DataType.EMBEDDING.dtype == np.ndarray
    assert DataType.PERTURBATION.dtype == pd.DataFrame


def test_datatype_properties():
    assert DataType.METADATA.is_input
    assert not DataType.METADATA.is_output
    assert not DataType.EMBEDDING.is_input
    assert DataType.EMBEDDING.is_output
    assert DataType.ANNDATA.description == "AnnData object containing expression data"


def test_datatype_anndata(example_anndata):
    assert isinstance(example_anndata, ad.AnnData)
    assert example_anndata.shape[0] > 0
    assert example_anndata.shape[1] > 0
