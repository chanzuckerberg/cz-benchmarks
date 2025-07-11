from pathlib import Path
import pandas as pd
import pytest

from czbenchmarks.datasets.single_cell import SingleCellDataset
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from czbenchmarks.datasets.types import Organism
from tests.datasets.test_single_cell_dataset import SingleCellDatasetTests
from tests.utils import create_dummy_anndata

class TestSingleCellLabeledDataset(SingleCellDatasetTests):

    @pytest.fixture
    def valid_dataset(self, tmp_path) -> SingleCellDataset:
        """Fixture to provide a valid SingleCellLabeledDataset H5AD file with valid human gene names and organism."""
        return SingleCellLabeledDataset(path=self.valid_dataset_file(tmp_path), organism=Organism.HUMAN)

    def valid_dataset_file(self, tmp_path) -> Path:
        """Fixture to provide a valid SingleCellLabeledDataset H5AD file with valid human gene names and organism."""
        
        file_path = tmp_path / "dummy.h5ad"
        adata = create_dummy_anndata(
            n_cells=5,
            n_genes=3,
            obs_columns=["cell_type"],
            organism=Organism.HUMAN,
        )
        adata.write_h5ad(file_path)

        return file_path

    def test_single_cell_labeled_dataset_validate_missing_cell_type(self, valid_dataset):
        """Test that SingleCellDataset validation fails when cell_type is missing."""
        valid_dataset.load_data()
        
        valid_dataset.adata.obs.drop(columns=["cell_type"], inplace=True)
        invalid_dataset = valid_dataset

        with pytest.raises(ValueError, match="Dataset does not contain 'cell_type' column in obs."):
            invalid_dataset.validate()


    def test_single_cell_labeled_dataset_store_task_inputs(self, tmp_path, valid_dataset):
        """Tests that the store_task_inputs method writes labels to a file."""
        valid_dataset.load_data()
        
        valid_dataset.store_task_inputs()

        output_file = tmp_path / "dummy_task_inputs" / "single_cell_labeled" / "labels_cell_type.json"
        assert output_file.exists()
        cell_types = pd.read_json(output_file, typ="series")
        assert not cell_types.empty
        assert all(cell_type in cell_types.values for cell_type in ["type_0", "type_1", "type_2"])

    @pytest.fixture
    def custom_label_dataset_file(self, tmp_path) -> Path:
        """Fixture to provide a SingleCellLabeledDataset H5AD file with custom label column."""
        
        file_path = tmp_path / "dummy_custom_label.h5ad"
        adata = create_dummy_anndata(
            n_cells=5,
            n_genes=3,
            obs_columns=["annotation"],
            organism=Organism.HUMAN,
        )
        adata.write_h5ad(file_path)

        return file_path

    def test_single_cell_labeled_dataset_custom_label_column_key(self, custom_label_dataset_file):
        """Test that SingleCellLabeledDataset uses custom label_column_key parameter."""
        dataset_file = custom_label_dataset_file
        
        # Test with custom label column key
        dataset = SingleCellLabeledDataset(
            path=dataset_file, 
            organism=Organism.HUMAN, 
            label_column_key="annotation"
        )
        
        dataset.load_data()
        
        # Verify that the labels are extracted from the custom column
        assert dataset.label_column_key == "annotation"
        assert "annotation" in dataset.adata.obs.columns
        assert len(dataset.labels) == 5
        assert all(label.startswith("annotation_") for label in dataset.labels)
        
        # Test validation passes with custom column
        dataset.validate()
        
        # Test that store_task_inputs creates file with custom label column name
        dataset.store_task_inputs()
        output_file = dataset.task_inputs_dir / "labels_annotation.json"
        assert output_file.exists()
        labels = pd.read_json(output_file, typ="series")
        assert not labels.empty
        assert all(label.startswith("annotation_") for label in labels.values)
        
        # Test that validation fails when custom column is missing
        dataset.adata.obs.drop(columns=["annotation"], inplace=True)
        with pytest.raises(ValueError, match="Dataset does not contain 'annotation' column in obs."):
            dataset.validate()
        
