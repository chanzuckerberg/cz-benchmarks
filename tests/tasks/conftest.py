import pytest
import numpy as np
import tempfile
import os
import pandas as pd
from czbenchmarks.datasets import (
    SingleCellDataset,
    PerturbationSingleCellDataset,
)
from czbenchmarks.datasets.types import Organism
from tests.utils import create_dummy_anndata


@pytest.fixture
def dummy_single_cell_dataset(n_cells: int = 1000, n_genes: int = 500):
    """Create a dummy dataset with 100 cells and 500 genes."""
    # Create a temporary directory and file
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "dummy.h5ad")

        # Create the dataset with the temporary file path
        dataset = SingleCellDataset(path=tmp_file, organism=Organism.HUMAN)
        adata = create_dummy_anndata(
            n_cells=n_cells,
            n_genes=n_genes,
            obs_columns=["cell_type", "batch"],
            var_columns=["feature_id"],
            organism=Organism.HUMAN,
        )

        # Write the AnnData object to the temporary file
        adata.write_h5ad(tmp_file)
        dataset.load_data()
        yield dataset



@pytest.fixture
def dummy_cross_species_datasets(n_cells: int = 500):
    """Create dummy datasets from different organisms for cross-species testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        datasets = []
        # Create one dataset for each organism
        for organism in [Organism.HUMAN, Organism.MOUSE]:
            tmp_file = os.path.join(tmp_dir, f"dummy_{organism.name.lower()}.h5ad")
            dataset = SingleCellDataset(path=tmp_file, organism=organism)

            # Create dummy data
            adata = create_dummy_anndata(
                n_cells=n_cells,
                n_genes=200,
                obs_columns=["cell_type"],
                organism=organism,
            )
            adata.write_h5ad(tmp_file)
            dataset.load_data()
            datasets.append(dataset)

        yield datasets


@pytest.fixture
def dummy_perturbation_dataset(
    n_cells: int = 500, condition_key: str = "condition", split_key: str = "split"
):
    """Create a dummy perturbation dataset with control and perturbed cells."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "dummy_perturbation.h5ad")
        dataset = PerturbationSingleCellDataset(
            path=tmp_file,
            organism=Organism.HUMAN,
            condition_key=condition_key,
            split_key=split_key,
        )

        # Create dummy data with control and perturbed cells
        n_genes = 200
        n_ctrl = n_cells // 2
        n_pert = n_cells - n_ctrl

        # Create base anndata with all cells
        adata = create_dummy_anndata(
            n_cells=n_cells,
            n_genes=n_genes,
            obs_columns=[condition_key, split_key],
            organism=Organism.HUMAN,
        )

        # Set condition and split annotations
        adata.obs[condition_key] = ["ctrl"] * n_ctrl + ["ENSG00000123456+ctrl"] * n_pert
        adata.obs[split_key] = ["train"] * n_ctrl + ["test"] * n_pert

        adata.write_h5ad(tmp_file)
        dataset.load_data()

        # Create and set predicted perturbation effect
        # pert_pred = pd.DataFrame(
        #     data=np.random.normal(size=(n_ctrl, n_genes)),
        #     columns=adata.var_names,
        #     index=adata[adata.obs[condition_key] == "ctrl"].obs_names,
        # )

        # dataset.set_output(
        #     ModelType.SCGENEPT,
        #     DataType.PERTURBATION_PRED,
        #     ("ENSG00000123456+ctrl", pert_pred),
        # )

        yield dataset