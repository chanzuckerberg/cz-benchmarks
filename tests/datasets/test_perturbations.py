import pytest
import numpy as np
import pandas as pd
import anndata as ad
from czbenchmarks.datasets.single_cell import PerturbationSingleCellDataset
from czbenchmarks.datasets.types import Organism, DataType

# Tests the loading of perturbation dataset data.
def test_perturbation_dataset_load_data(tmp_path):
    path = tmp_path / "perturbation.h5ad"
    X = np.ones((6, 3))
    obs = pd.DataFrame({
        "condition": ["ctrl","ctrl","test1","test1","test2","test2"],
        "split": ["train","train","test","test","test","test"]
    }, index=[f"cell_{i}" for i in range(6)])
    var = pd.DataFrame(index=["ENSG1","ENSG2","ENSG3"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(str(path))
    ds = PerturbationSingleCellDataset(str(path), Organism.HUMAN, condition_key="condition", split_key="split")
    ds.load_data()
    truth = ds.get_input(DataType.PERTURBATION_TRUTH)
    assert "test1" in truth
    assert "test2" in truth
    assert ds.adata.shape == (2, 3)

# Tests that loading data fails when the condition key is missing.
def test_perturbation_dataset_load_data_missing_condition_key(tmp_path):
    path = tmp_path / "perturbation_missing_condition.h5ad"
    X = np.ones((6, 3))
    obs = pd.DataFrame({
        "cond": ["ctrl","ctrl","test1","test1","test2","test2"],
        "split": ["train","train","test","test","test","test"]
    }, index=[f"cell_{i}" for i in range(6)])
    var = pd.DataFrame(index=["ENSG1","ENSG2","ENSG3"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(str(path))
    ds = PerturbationSingleCellDataset(str(path), Organism.HUMAN, condition_key="condition", split_key="split")
    with pytest.raises(AssertionError):
        ds.load_data()

# Tests the unloading of perturbation dataset data.
def test_perturbation_dataset_unload_data(tmp_path):
    path = tmp_path / "perturbation_unload.h5ad"
    X = np.ones((6, 3))
    obs = pd.DataFrame({
        "condition": ["ctrl","ctrl","test1","test1","test2","test2"],
        "split": ["train","train","test","test","test","test"]
    }, index=[f"cell_{i}" for i in range(6)])
    var = pd.DataFrame(index=["ENSG1","ENSG2","ENSG3"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(str(path))
    ds = PerturbationSingleCellDataset(str(path), Organism.HUMAN, condition_key="condition", split_key="split")
    ds.load_data()
    ds.unload_data()
    with pytest.raises(KeyError):
        ds.get_input(DataType.PERTURBATION_TRUTH)

# Tests the properties of the PerturbationSingleCellDataset class.
def test_perturbation_dataset_properties(tmp_path):
    path = tmp_path / "perturbation_props.h5ad"
    X = np.ones((4, 2))
    obs = pd.DataFrame({
        "condition": ["ctrl","test","ctrl","test"],
        "split": ["train","test","train","test"]
    }, index=[f"cell_{i}" for i in range(4)])
    var = pd.DataFrame(index=["ENSG1","ENSG2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.write_h5ad(str(path))
    ds = PerturbationSingleCellDataset(str(path), Organism.HUMAN, condition_key="cond", split_key="spl")
    ds.set_input(DataType.CONDITION_KEY, "cond")
    ds.set_input(DataType.SPLIT_KEY, "spl")
    assert ds.condition_key == "cond"
    assert ds.split_key == "spl"
