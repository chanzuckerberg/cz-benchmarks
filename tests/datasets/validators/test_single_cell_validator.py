from czbenchmarks.datasets.validators.single_cell_validator import (
    SingleCellLabeledValidator,
)
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from czbenchmarks.datasets.types import Organism


class TestSingleCellLabeledValidator:
    """Tests for SingleCellLabeledValidator class."""

    class ConcreteValidator(SingleCellLabeledValidator):
        available_organisms = [Organism.HUMAN]
        required_obs_keys = ["cell_type"]
        required_var_keys = ["gene_name"]

    def test_concrete_validator_instantiation(self):
        """Test that a valid subclass of SingleCellLabeledValidator can be instantiated."""
        validator = self.ConcreteValidator()
        assert isinstance(validator, SingleCellLabeledValidator)

    def test_validate_dataset(self, tmp_path):
        """Test validation logic for a valid dataset."""
        dummy_path = tmp_path / "dummy.h5ad"
        dummy_path.touch()

        import anndata
        import numpy as np

        import pandas as pd

        adata = anndata.AnnData(
            X=np.zeros((1, 1)),
            obs=pd.DataFrame({"cell_type": ["typeA"]}),
            var=pd.DataFrame({"gene_name": ["geneA"]}),
        )

        dataset = SingleCellLabeledDataset(
            path=dummy_path,
            organism=Organism.HUMAN,
            label_column_key="cell_type",
        )
        dataset.adata = adata

        validator = self.ConcreteValidator()

        validator._validate_dataset(dataset)
        assert True
