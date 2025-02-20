import anndata as ad
import numpy as np
import pandas as pd
import pytest
from czibench.datasets.base import BaseDataset
from czibench.datasets.sc import SingleCellDataset
from czibench.datasets.types import DataType, Organism

TEST_H5AD_PATH = "tests/assets/example-small.h5ad"


class DummyDataset(BaseDataset):
    def _validate(self) -> None:
        if not hasattr(self, "path"):
            raise ValueError("Path attribute is missing.")

    def load_data(self) -> None:
        self.set_input(DataType.METADATA, pd.DataFrame({"test": [1, 2, 3]}))

    def unload_data(self) -> None:
        self._inputs.pop(DataType.METADATA, None)


@pytest.fixture
def dummy_dataset(tmp_path):
    file_path = tmp_path / "dummy_file.txt"
    file_path.write_text("dummy content")
    ds = DummyDataset(path=str(file_path))
    return ds


@pytest.fixture
def invalid_dummy_dataset():
    ds = DummyDataset(path="non_existent_file.txt")
    return ds


@pytest.fixture
def single_cell_dataset_human():
    ds = SingleCellDataset(path=TEST_H5AD_PATH, organism=Organism.HUMAN)
    return ds


@pytest.fixture
def invalid_anndata_file(tmp_path):
    obs_data = pd.DataFrame({"cell_type": ["X", "Y"]}, index=["cell1", "cell2"])
    var_data = pd.DataFrame(
        {"gene_symbol": ["ABC0001", "ABC0002"]}, index=["ABC0001", "ABC0002"]
    )
    adata = ad.AnnData(X=np.random.rand(2, 2), obs=obs_data, var=var_data)
    file_path = tmp_path / "invalid_data.h5ad"
    adata.write_h5ad(file_path)
    return file_path


@pytest.fixture
def mouse_anndata_file(tmp_path):
    obs_data = pd.DataFrame({"cell_type": ["A", "B"]}, index=["cell1", "cell2"])
    var_data = pd.DataFrame(
        {"ensembl_id": ["ENSMUSG0000000001", "ENSMUSG0000000002"]},
        index=["gene1", "gene2"],
    )
    adata = ad.AnnData(X=np.random.rand(2, 2), obs=obs_data, var=var_data)
    file_path = tmp_path / "mouse_data.h5ad"
    adata.write_h5ad(file_path)
    return file_path


def test_dummy_dataset_valid(dummy_dataset):
    dummy_dataset.validate()
    dummy_dataset.load_data()
    assert DataType.METADATA in dummy_dataset.inputs
    dummy_dataset.unload_data()
    assert DataType.METADATA not in dummy_dataset.inputs


def test_dummy_dataset_invalid_path(invalid_dummy_dataset):
    with pytest.raises(ValueError, match="is not valid"):
        invalid_dummy_dataset.validate()


def test_dummy_dataset_wrong_input_type(dummy_dataset):
    with pytest.raises(TypeError):
        dummy_dataset.set_input(DataType.ANNDATA, pd.DataFrame())


def test_dummy_dataset_serialize_deserialize(dummy_dataset, tmp_path):
    dummy_dataset.validate()
    dummy_dataset.load_data()
    out_file = tmp_path / "dummy_dataset_serialized"
    dummy_dataset.serialize(str(out_file))
    loaded = BaseDataset.deserialize(str(out_file))
    assert isinstance(loaded, DummyDataset)
    assert DataType.METADATA in loaded.inputs


def test_single_cell_dataset_load_data(single_cell_dataset_human):
    single_cell_dataset_human.load_data()
    assert DataType.ANNDATA in single_cell_dataset_human.inputs
    assert DataType.METADATA in single_cell_dataset_human.inputs
    single_cell_dataset_human.unload_data()
    assert DataType.ANNDATA not in single_cell_dataset_human.inputs
    assert DataType.METADATA not in single_cell_dataset_human.inputs


def test_single_cell_dataset_validate_success(single_cell_dataset_human):
    single_cell_dataset_human.load_data()
    single_cell_dataset_human.validate()


def test_single_cell_dataset_no_anndata():
    ds = SingleCellDataset(path="non_existent_file.h5ad", organism=Organism.HUMAN)
    with pytest.raises(ValueError, match="Dataset non_existent_file.h5ad is not valid"):
        ds.validate()


def test_single_cell_dataset_invalid_gene_names(invalid_anndata_file):
    ds = SingleCellDataset(path=str(invalid_anndata_file), organism=Organism.HUMAN)
    ds.load_data()
    with pytest.raises(ValueError, match="valid gene names"):
        ds.validate()


def test_single_cell_dataset_invalid_prefix(invalid_anndata_file):
    ds = SingleCellDataset(path=str(invalid_anndata_file), organism=Organism.HUMAN)
    ds.load_data()
    with pytest.raises(ValueError, match="valid gene names"):
        ds.validate()


def test_single_cell_dataset_serialize_deserialize(single_cell_dataset_human, tmp_path):
    single_cell_dataset_human.load_data()
    single_cell_dataset_human.validate()
    out_file = tmp_path / "sc_dataset_serialized"
    single_cell_dataset_human.serialize(str(out_file))
    loaded = SingleCellDataset.deserialize(str(out_file))
    assert isinstance(loaded, SingleCellDataset)
    loaded.load_data()
    loaded.validate()


def test_single_cell_dataset_invalid_path():
    ds = SingleCellDataset(path="does_not_exist.h5ad", organism=Organism.HUMAN)
    with pytest.raises(ValueError, match="is not valid"):
        ds.validate()


def test_single_cell_dataset_relabel_var_names_with_ensembl_id(mouse_anndata_file):
    ds = SingleCellDataset(path=str(mouse_anndata_file), organism=Organism.MOUSE)
    ds.load_data()
    ds.validate()


def test_single_cell_dataset_set_output(single_cell_dataset_human):
    single_cell_dataset_human.load_data()
    single_cell_dataset_human.validate()
    single_cell_dataset_human.set_output(DataType.EMBEDDING, np.random.rand(10, 5))
    assert DataType.EMBEDDING in single_cell_dataset_human.outputs


def test_single_cell_dataset_organism_mismatch(mouse_anndata_file):
    ds = SingleCellDataset(path=str(mouse_anndata_file), organism=Organism.HUMAN)
    ds.load_data()
    with pytest.raises(ValueError, match="valid gene names"):
        ds.validate()


def test_organism_from_string():
    assert Organism.from_string("human") == Organism.HUMAN
    assert Organism.from_string("mouse") == Organism.MOUSE
    with pytest.raises(ValueError, match="Unknown organism"):
        Organism.from_string("unknown")
