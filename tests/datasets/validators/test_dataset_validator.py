import pytest
from czbenchmarks.datasets.dataset import Dataset
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from czbenchmarks.datasets.validators.base_dataset_validator import BaseDatasetValidator
from czbenchmarks.datasets import Organism
from czbenchmarks.datasets.validators.base_single_cell_validator import (
    BaseSingleCellLabeledValidator,
)

import anndata as ad
import numpy as np
import pandas as pd
from typing import ClassVar, List, Type


# Mock Dataset Class for testing BaseDatasetValidator
class MockDataset(Dataset):
    """A mock dataset for testing the base validator."""

    def load_data(self) -> None:
        pass
    
    def store_task_inputs(self):
        pass

    def _validate(self) -> None:
        pass



# Test Cases for BaseDatasetValidator
class TestDatasetValidator:
    def test_successful_instantiation(self):
        """Tests that a correctly defined subclass can be instantiated."""

        class ConcreteValidator(BaseDatasetValidator):
            dataset_type: ClassVar[Type[Dataset]] = MockDataset

            def _validate_dataset(self, dataset: Dataset):
                pass

            @property
            def inputs(self):
                return {"X"}

            @property
            def outputs(self):
                return {"y"}

        validator = ConcreteValidator()
        assert isinstance(validator, BaseDatasetValidator)

    def test_missing_dataset_type(self):
        """Tests that a TypeError is raised if dataset_type is missing."""
        with pytest.raises(TypeError, match="without dataset_type class variable"):

            class InvalidValidator(BaseDatasetValidator):
                def _validate_dataset(self, dataset: Dataset):
                    pass

                @property
                def inputs(self):
                    return {"X"}

                @property
                def outputs(self):
                    return {"y"}

    def test_dataset_type_mismatch(self, tmp_path):
        """Tests that a ValueError is raised for dataset type mismatch."""

        class ConcreteValidator(BaseDatasetValidator):
            dataset_type: ClassVar[Type[Dataset]] = MockDataset

            def _validate_dataset(self, dataset: Dataset):
                pass

            @property
            def inputs(self):
                return {"X"}

            @property
            def outputs(self):
                return {"y"}

        validator = ConcreteValidator()
        # Use the actual SingleCellDataset which will cause a type mismatch
        wrong_dataset = SingleCellLabeledDataset(path=str(tmp_path), organism=Organism.HUMAN)

        with pytest.raises(ValueError, match="Dataset type mismatch"):
            validator.validate_dataset(wrong_dataset)

    def test_successful_validation(self, tmp_path):
        """Tests that validation succeeds with a valid dataset."""

        class ConcreteValidator(BaseDatasetValidator):
            dataset_type: ClassVar[Type[Dataset]] = MockDataset

            def _validate_dataset(self, dataset: Dataset):
                pass

        validator = ConcreteValidator()
        dataset = MockDataset(dataset_type_name="mock", path=str(tmp_path), organism=Organism.HUMAN)

        try:
            validator.validate_dataset(dataset)
        except ValueError:
            pytest.fail("validate_dataset raised ValueError unexpectedly!")


# Test Cases for BaseSingleCellValidator
class TestBaseSingleCellLabeledValidator:
    @pytest.fixture
    def sc_dataset(self, tmp_path) -> SingleCellLabeledDataset:
        """Fixture to create a valid SingleCellLabeledDataset instance for testing."""
        # Create a dummy dataset instance.
        # We manually assign adata to avoid file I/O.
        dataset = SingleCellLabeledDataset(path=str(tmp_path), organism=Organism.HUMAN)
        dataset.adata = ad.AnnData(
            X=np.random.poisson(size=(10, 10)),
            obs=pd.DataFrame(index=[f"cell_{i}" for i in range(10)]),
            var=pd.DataFrame(
                index=[f"{Organism.HUMAN.prefix}_gene_{i}" for i in range(10)]
            ),
        )
        # Monkeypatch the 'inputs' property, which is required by the validator.
        # It must be a dict to work with the `.keys()` call in the source.
        dataset.inputs = {"X": True}
        return dataset

    def test_successful_instantiation_sc(self):
        """Tests successful instantiation of a BaseSingleCellValidator subclass."""

        class ConcreteSCValidator(BaseSingleCellLabeledValidator):
            available_organisms: ClassVar[List[Organism]] = [Organism.HUMAN]
            required_obs_keys: ClassVar[List[str]] = []
            required_var_keys: ClassVar[List[str]] = []

            @property
            def inputs(self):
                return {"X"}

            @property
            def outputs(self):
                return set()

        validator = ConcreteSCValidator()
        assert isinstance(validator, BaseSingleCellLabeledValidator)

    def test_missing_available_organisms(self):
        """Tests TypeError for missing available_organisms."""
        with pytest.raises(
            TypeError, match="without available_organisms class variable"
        ):

            class InvalidSCValidator(BaseSingleCellLabeledValidator):
                required_obs_keys: ClassVar[List[str]] = []
                required_var_keys: ClassVar[List[str]] = []

                @property
                def inputs(self):
                    return set()

                @property
                def outputs(self):
                    return set()

    def test_missing_required_obs_keys(self):
        """Tests TypeError for missing required_obs_keys."""
        with pytest.raises(TypeError, match="without required_obs_keys class variable"):

            class InvalidSCValidator(BaseSingleCellLabeledValidator):
                available_organisms: ClassVar[List[Organism]] = [Organism.HUMAN]
                required_var_keys: ClassVar[List[str]] = []

                @property
                def inputs(self):
                    return set()

                @property
                def outputs(self):
                    return set()

    def test_missing_required_var_keys(self):
        """Tests TypeError for missing required_var_keys."""
        with pytest.raises(TypeError, match="without required_var_keys class variable"):

            class InvalidSCValidator(BaseSingleCellLabeledValidator):
                available_organisms: ClassVar[List[Organism]] = [Organism.HUMAN]
                required_obs_keys: ClassVar[List[str]] = []

                @property
                def inputs(self):
                    return set()

                @property
                def outputs(self):
                    return set()

    def test_unsupported_organism(self, sc_dataset):
        """Tests ValueError for an unsupported organism."""

        class ConcreteSCValidator(BaseSingleCellLabeledValidator):
            available_organisms: ClassVar[List[Organism]] = [
                Organism.MOUSE
            ]  # Expects MOUSE
            required_obs_keys: ClassVar[List[str]] = []
            required_var_keys: ClassVar[List[str]] = []

            @property
            def inputs(self):
                return {"X"}

            @property
            def outputs(self):
                return set()

        validator = ConcreteSCValidator()
        sc_dataset.organism = Organism.HUMAN  # Provide HUMAN

        with pytest.raises(ValueError, match="is not supported"):
            validator.validate_dataset(sc_dataset)

    def test_missing_obs_keys(self, sc_dataset):
        """Tests ValueError for missing required observation keys."""

        class ConcreteSCValidator(BaseSingleCellLabeledValidator):
            available_organisms: ClassVar[List[Organism]] = [Organism.HUMAN]
            required_obs_keys: ClassVar[List[str]] = ["batch"]  # Requires 'batch'
            required_var_keys: ClassVar[List[str]] = []

            @property
            def inputs(self):
                return {"X"}

            @property
            def outputs(self):
                return set()

        validator = ConcreteSCValidator()

        with pytest.raises(
            ValueError, match="Missing required obs keys: \\['batch'\\]"
        ):
            validator.validate_dataset(sc_dataset)

    def test_missing_var_keys(self, sc_dataset):
        """Tests ValueError for missing required variable keys."""

        class ConcreteSCValidator(BaseSingleCellLabeledValidator):
            available_organisms: ClassVar[List[Organism]] = [Organism.HUMAN]
            required_obs_keys: ClassVar[List[str]] = []
            required_var_keys: ClassVar[List[str]] = [
                "gene_symbol"
            ]  # Requires 'gene_symbol'

            @property
            def inputs(self):
                return {"X"}

            @property
            def outputs(self):
                return set()

        validator = ConcreteSCValidator()

        with pytest.raises(
            ValueError, match="Missing required var keys: \\['gene_symbol'\\]"
        ):
            validator.validate_dataset(sc_dataset)

    def test_successful_sc_validation(self, sc_dataset):
        """Tests successful validation of a single-cell dataset."""

        class ConcreteSCValidator(BaseSingleCellLabeledValidator):
            available_organisms: ClassVar[List[Organism]] = [Organism.HUMAN]
            required_obs_keys: ClassVar[List[str]] = ["batch"]
            required_var_keys: ClassVar[List[str]] = ["gene_symbol"]

            @property
            def inputs(self):
                return {"X"}

            @property
            def outputs(self):
                return set()

        validator = ConcreteSCValidator()
        # Add the required keys to the dataset's anndata object
        sc_dataset.adata.obs["batch"] = "A"
        sc_dataset.adata.var["gene_symbol"] = "GENE"

        try:
            validator.validate_dataset(sc_dataset)
        except ValueError:
            pytest.fail("Single-cell validation failed unexpectedly!")
