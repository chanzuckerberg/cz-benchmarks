import pytest
from czbenchmarks.datasets.single_cell import PerturbationSingleCellDataset
from czbenchmarks.datasets.types import Organism, DataType
from tests.utils import create_dummy_anndata


def test_perturbation_dataset_load_data(dummy_perturbation_anndata):
    """Tests the loading of perturbation dataset data."""
    ds = PerturbationSingleCellDataset(
        dummy_perturbation_anndata,
        Organism.HUMAN,
        condition_key="condition",
        split_key="split",
    )
    ds.load_data()
    truth = ds.get_input(DataType.PERTURBATION_TRUTH)
    assert "test1" in truth
    assert "test2" in truth
    assert ds.adata.shape == (2, 3)


def test_perturbation_dataset_load_data_missing_condition_key(tmp_path):
    """Tests that loading data fails when the condition key is missing."""
    path = tmp_path / "perturbation_missing_condition.h5ad"
    adata = create_dummy_anndata(
        n_cells=6, n_genes=3, obs_columns=["split"], organism=Organism.HUMAN
    )
    adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
    adata.write_h5ad(str(path))

    ds = PerturbationSingleCellDataset(
        str(path), Organism.HUMAN, condition_key="condition", split_key="split"
    )
    with pytest.raises(AssertionError):
        ds.load_data()


def test_perturbation_dataset_load_data_missing_split_key(tmp_path):
    """Tests that loading data fails when the split key is missing."""
    path = tmp_path / "perturbation_missing_split.h5ad"
    adata = create_dummy_anndata(
        n_cells=6, n_genes=3, obs_columns=["condition"], organism=Organism.HUMAN
    )
    adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
    adata.write_h5ad(str(path))

    ds = PerturbationSingleCellDataset(
        str(path), Organism.HUMAN, condition_key="condition", split_key="split"
    )
    with pytest.raises(AssertionError):
        ds.load_data()


def test_perturbation_dataset_validate_invalid_split(tmp_path):
    """Tests that validation fails when split column contains invalid literals."""
    path = tmp_path / "perturbation_invalid_split.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition", "split"],
        organism=Organism.HUMAN,
    )
    adata.obs["condition"] = ["ctrl", "ctrl", "test1", "test1", "test2", "test2"]
    adata.obs["split"] = ["invalid", "train", "test", "test", "test", "test"]
    adata.write_h5ad(str(path))

    ds = PerturbationSingleCellDataset(
        str(path), Organism.HUMAN, condition_key="condition", split_key="split"
    )
    ds.load_data()
    with pytest.raises(ValueError, match="Invalid split value"):
        ds.validate()


def test_perturbation_dataset_validate_invalid_condition_format(tmp_path):
    """Tests that validation fails when condition column format is invalid."""
    path = tmp_path / "perturbation_invalid_condition.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition", "split"],
        organism=Organism.HUMAN,
    )
    adata.obs["condition"] = [
        "control",
        "control",
        "test1",
        "test1",
        "test2",
        "test2",
    ]  # "control" instead of "ctrl"
    adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
    adata.write_h5ad(str(path))

    ds = PerturbationSingleCellDataset(
        str(path), Organism.HUMAN, condition_key="condition", split_key="split"
    )
    ds.load_data()
    with pytest.raises(ValueError, match="Invalid control condition"):
        ds.validate()


def test_perturbation_dataset_validate_invalid_perturbation_format(tmp_path):
    """Tests that validation fails when perturbation condition format is invalid."""
    path = tmp_path / "perturbation_invalid_pert.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition", "split"],
        organism=Organism.HUMAN,
    )
    # Test invalid format (no + separator)
    adata.obs["condition"] = [
        "ctrl",
        "ctrl",
        "ENSG00000123456",  # Missing + separator
        "ENSG00000123456",
        "test2",
        "test2",
    ]
    adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
    adata.write_h5ad(str(path))

    ds = PerturbationSingleCellDataset(
        str(path), Organism.HUMAN, condition_key="condition", split_key="split"
    )
    ds.load_data()
    with pytest.raises(ValueError, match="Invalid perturbation condition format"):
        ds.validate()


def test_perturbation_dataset_validate_invalid_gene_prefix(tmp_path):
    """Tests that validation fails when gene prefix in condition is invalid."""
    path = tmp_path / "perturbation_invalid_gene.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition", "split"],
        organism=Organism.HUMAN,
    )
    # Test invalid gene prefix
    adata.obs["condition"] = [
        "ctrl",
        "ctrl",
        "INVALID123+ctrl",  # Wrong prefix
        "INVALID123+ctrl",
        "test2+ctrl",
        "test2+ctrl",
    ]
    adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
    adata.write_h5ad(str(path))

    ds = PerturbationSingleCellDataset(
        str(path), Organism.HUMAN, condition_key="condition", split_key="split"
    )
    ds.load_data()
    with pytest.raises(ValueError, match="Invalid gene prefix in condition"):
        ds.validate()


def test_perturbation_dataset_validate_invalid_combo_prefix(tmp_path):
    """Tests that validation fails when gene prefix in combinatorial perturbation is invalid."""
    path = tmp_path / "perturbation_invalid_combo.h5ad"
    adata = create_dummy_anndata(
        n_cells=6,
        n_genes=3,
        obs_columns=["condition", "split"],
        organism=Organism.HUMAN,
    )
    # Test invalid gene prefix in combinatorial perturbation
    adata.obs["condition"] = [
        "ctrl",
        "ctrl",
        "ENSG00000123456+INVALID123",  # Second gene has wrong prefix
        "ENSG00000123456+INVALID123",
        "test2+ctrl",
        "test2+ctrl",
    ]
    adata.obs["split"] = ["train", "train", "test", "test", "test", "test"]
    adata.write_h5ad(str(path))

    ds = PerturbationSingleCellDataset(
        str(path), Organism.HUMAN, condition_key="condition", split_key="split"
    )
    ds.load_data()
    with pytest.raises(ValueError, match="Invalid gene prefix in condition"):
        ds.validate()


def test_perturbation_dataset_validate_valid_conditions(tmp_path):
    """Tests that validation passes with valid condition formats."""
    path = tmp_path / "perturbation_valid.h5ad"
    adata = create_dummy_anndata(
        n_cells=9,
        n_genes=3,
        obs_columns=["condition", "split"],
        organism=Organism.HUMAN,
    )
    # Test all valid formats
    adata.obs["condition"] = [
        "ctrl",
        "ctrl",
        "ENSG00000123456+ctrl",  # Single gene perturbation
        "ENSG00000123456+ctrl",
        "ENSG00000123456+ENSG00000789012",  # Combinatorial perturbation
        "ENSG00000123456+ENSG00000789012",
        "ENSG00000111111+ctrl",
        "ENSG00000111111+ctrl",
        "ctrl",
    ]
    adata.obs["split"] = ["train"] * 3 + ["test"] * 6
    adata.write_h5ad(str(path))

    ds = PerturbationSingleCellDataset(
        str(path), Organism.HUMAN, condition_key="condition", split_key="split"
    )
    ds.load_data()
    try:
        ds.validate()
    except Exception as e:
        pytest.fail(f"Validation failed unexpectedly: {e}")


def test_perturbation_dataset_unload_data(dummy_perturbation_anndata):
    """Tests the unloading of perturbation dataset data."""
    ds = PerturbationSingleCellDataset(
        dummy_perturbation_anndata,
        Organism.HUMAN,
        condition_key="condition",
        split_key="split",
    )
    ds.load_data()
    ds.unload_data()
    with pytest.raises(KeyError):
        ds.get_input(DataType.PERTURBATION_TRUTH)


def test_perturbation_dataset_properties(dummy_perturbation_anndata):
    """Tests the properties of the PerturbationSingleCellDataset class."""
    ds = PerturbationSingleCellDataset(
        dummy_perturbation_anndata,
        Organism.HUMAN,
        condition_key="cond",
        split_key="spl",
    )
    ds.set_input(DataType.CONDITION_KEY, "cond")
    ds.set_input(DataType.SPLIT_KEY, "spl")
    assert ds.condition_key == "cond"
    assert ds.split_key == "spl"
