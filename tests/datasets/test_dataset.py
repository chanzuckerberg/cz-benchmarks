from abc import abstractmethod
from pathlib import Path
import pytest

from czbenchmarks.datasets.dataset import Dataset


class DatasetTests:
    """Tests for Dataset class that should be tested for each concrete subclass. This is accomplished by having the concrete subclass extend this class and implement the `valid_dataset` fixture, which returns a valid instance of the dataset class, used by the tests herein."""

    @pytest.fixture
    @abstractmethod
    def valid_dataset(self, tmp_path) -> Dataset:
        pass

    def test_single_cell_dataset_validate_success(self, valid_dataset):
        """Test that Dataset load and validation succeeds for a valid dataset."""
        # Should not raise any exceptions
        valid_dataset.load_data()

        # Should not raise any exceptions
        valid_dataset.validate()

    def test_validate_dataset_wrong_organism_type(self, tmp_path, valid_dataset):
        """Tests that dataset validation fails when the organism type is invalid."""
        valid_dataset.organism = "invalid_organism"
        invalid_dataset = valid_dataset

        with pytest.raises(ValueError, match="Organism is not a valid Organism enum"):
            invalid_dataset.validate()

    def test_custom_input_tasks_dir_is_used(self, tmp_path, valid_dataset):
        """Test that custom task inputs directory is used if provided."""
        custom_task_inputs_dir = tmp_path / "custom_task_inputs"
        valid_dataset.task_inputs_dir = custom_task_inputs_dir

        valid_dataset.load_data()
        valid_dataset.store_task_inputs()

        assert custom_task_inputs_dir.exists()
        assert len(list(custom_task_inputs_dir.iterdir())) > 0


class TestDataset:
    """Tests for Dataset class that cannot be easily tested for each concrete subclass."""

    class ConcreteDataset(Dataset):
        def __init__(self, path, organism, **kwargs):
            super().__init__(
                dataset_type_name="concrete_dataset",
                path=path,
                organism=organism,
                **kwargs,
            )

        def load_data(self):
            pass

        def store_task_inputs(self):
            pass

        def _validate(self):
            pass

    def test_validate_dataset_path_not_exists(self, tmp_path):
        """Test that validation fails when dataset path does not exist.
        Note that this cannot be tested with a concrete dataset class."""

        with pytest.raises(ValueError, match="Dataset path does not exist"):
            TestDataset.ConcreteDataset(
                path=tmp_path / "non_existent_path", organism="HUMAN"
            )

    def test_task_inputs_dir_correct(self, tmp_path):
        """Test that task inputs directory is correctly set based on the dataset path."""
        dataset = TestDataset.ConcreteDataset(path=tmp_path, organism="HUMAN")

        assert dataset.task_inputs_dir == Path(
            f"{tmp_path.with_suffix('')}_task_inputs/concrete_dataset"
        ), "Task inputs directory is not correctly set based on the dataset path."
