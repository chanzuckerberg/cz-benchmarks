import pytest
import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir,'..','src'))
sys.path.append(os.path.join(root_dir,'..','docker'))


from czibench.models.sc import BaseSingleCell
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import Organism
from czibench.models.base import BaseModel
from unittest.mock import MagicMock

class DummySingleCellModel(BaseSingleCell):
    available_organisms = [Organism.HUMAN]
    required_obs_keys = ["cell_type"]
    required_var_keys = ["gene_name"]

    @classmethod
    def _validate_model_requirements(cls, dataset: SingleCellDataset):
        if "gene_name" not in dataset.adata.var.columns:
            raise ValueError("Missing required var key: gene_name")

    def get_model_weights_subdir(self) -> str:
        return "dummy_weights"

    def _download_model_weights(self):
        pass

    def run_model(self):
        pass

def test_base_single_cell_subclass():
    """Test if a subclass of BaseSingleCell can be instantiated correctly."""
    model = DummySingleCellModel()
    assert isinstance(model, BaseSingleCell)
    assert issubclass(DummySingleCellModel, BaseModel)

def test_missing_class_variables():
    """Ensure that missing class variables raise TypeError."""
    with pytest.raises(TypeError):
        class InvalidModel(BaseSingleCell):
            pass

def test_validate_dataset_valid():
    """Test dataset validation with a valid dataset."""
    dataset = SingleCellDataset("dummy_path", Organism.HUMAN)
    dataset.adata = MagicMock()
    dataset.adata.var.columns = ["gene_name"]
    DummySingleCellModel._validate_dataset(dataset)  # Should not raise an error

# def test_validate_dataset_invalid_organism():
#     """Test that an unsupported organism raises a ValueError."""
#     dataset = SingleCellDataset("dummy_path", Organism.MOUSE)  # Unsupported organism

#     with pytest.raises(ValueError, match="Dataset organism mus_musculus is not supported for DummySingleCellModel"):
#         DummySingleCellModel._validate_dataset(dataset)

def test_validate_dataset_missing_var_key():
    """Test that missing required var keys raise a ValueError."""
    dataset = SingleCellDataset("dummy_path", Organism.HUMAN)
    dataset.adata = MagicMock()
    dataset.adata.var.columns = []  # Missing "gene_name"
    with pytest.raises(ValueError, match="Missing required var key: gene_name"):
        DummySingleCellModel._validate_dataset(dataset)
