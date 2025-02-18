import pytest
import anndata
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism
from czibench.models.validators.geneformer import GeneformerValidator
from czibench.models.validators.scgenept import ScGenePTValidator
from czibench.models.validators.scgpt import ScGPTValidator
from czibench.models.validators.scvi import SCVIValidator
from czibench.models.validators.uce import UCEValidator

@pytest.fixture
def example_adata():
    return anndata.read_h5ad("tests/assets/example-small.h5ad")

@pytest.fixture
def single_cell_dataset_human(example_adata):
    ds = SingleCellDataset("tests/assets/example-small.h5ad", organism=Organism.HUMAN)
    ds.adata = example_adata
    return ds

@pytest.fixture
def single_cell_dataset_mouse(example_adata):
    ds = SingleCellDataset("tests/assets/example-small.h5ad", organism=Organism.MOUSE)
    ds.adata = example_adata
    return ds

def test_geneformer_validator(single_cell_dataset_human):
    single_cell_dataset_human.adata.var["feature_id"] = "feature_x"
    validator = GeneformerValidator()
    validator.validate_dataset(single_cell_dataset_human)

def test_scgenept_validator(single_cell_dataset_human):
    single_cell_dataset_human.adata.var["gene_symbol"] = "gene_symbol_x"
    validator = ScGenePTValidator()
    validator.validate_dataset(single_cell_dataset_human)

def test_scgpt_validator(single_cell_dataset_human):
    single_cell_dataset_human.adata.var["gene_symbol"] = "gene_symbol_y"
    validator = ScGPTValidator()
    validator.validate_dataset(single_cell_dataset_human)

def test_scvi_validator_human(single_cell_dataset_human):
    single_cell_dataset_human.adata.obs["dataset_id"] = "test"
    single_cell_dataset_human.adata.obs["assay"] = "test"
    single_cell_dataset_human.adata.obs["suspension_type"] = "test"
    single_cell_dataset_human.adata.obs["donor_id"] = "test"
    validator = SCVIValidator()
    validator.validate_dataset(single_cell_dataset_human)

def test_scvi_validator_mouse(single_cell_dataset_mouse):
    single_cell_dataset_mouse.adata.obs["dataset_id"] = "test"
    single_cell_dataset_mouse.adata.obs["assay"] = "test"
    single_cell_dataset_mouse.adata.obs["suspension_type"] = "test"
    single_cell_dataset_mouse.adata.obs["donor_id"] = "test"
    validator = SCVIValidator()
    validator.validate_dataset(single_cell_dataset_mouse)

def test_uce_validator_human(single_cell_dataset_human):
    single_cell_dataset_human.adata.var["gene_symbol"] = "gene_symbol_x"
    validator = UCEValidator()
    validator.validate_dataset(single_cell_dataset_human)

def test_uce_validator_mouse(single_cell_dataset_mouse):
    single_cell_dataset_mouse.adata.var["gene_symbol"] = "gene_symbol_y"
    validator = UCEValidator()
    validator.validate_dataset(single_cell_dataset_mouse)
