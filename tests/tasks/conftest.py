import pytest
import pandas as pd
import numpy as np
import anndata as ad
from czbenchmarks.tasks.types import CellRepresentation
from czbenchmarks.datasets.types import Organism
from czbenchmarks.metrics.types import MetricResult
from tests.utils import create_dummy_anndata


@pytest.fixture
def dummy_anndata():
    n_cells: int = 500
    n_genes: int = 200
    organism: Organism = Organism.HUMAN
    obs_columns: list[str] = ["cell_type", "batch"]
    var_columns: list[str] = ["feature_name"]
    anndata: ad.AnnData = create_dummy_anndata(
        n_cells=n_cells,
        n_genes=n_genes,
        organism=organism,
        obs_columns=obs_columns,
        var_columns=var_columns,
    )

    expression_matrix: CellRepresentation = anndata.X.copy()
    obs: pd.DataFrame = anndata.obs.copy()
    var: pd.DataFrame = anndata.var.copy()

    # TODO perform PCA on expression matrix to get true embedding
    embedding_matrix: CellRepresentation = expression_matrix.toarray()

    return {
        "anndata": anndata,
        "expression_matrix": expression_matrix,
        "obs": obs,
        "var": var,
        "embedding_matrix": embedding_matrix,
    }


@pytest.fixture
def adata_uns_setup():
    """Helper function to setup common AnnData UNS structure."""

    def _setup_uns(adata, de_results, target_conditions_dict, control_cells_map):
        adata.uns.update(
            {
                "control_cells_ids": {},
                "de_results": de_results[["condition", "gene_id", "logfoldchange"]]
                if "gene_id" in de_results.columns
                else de_results,
                "metric_column": "logfoldchange",
                "target_conditions_dict": target_conditions_dict,
                "control_cells_map": control_cells_map,
            }
        )
        return adata

    return _setup_uns


@pytest.fixture
def assert_metric_results():
    """Helper function for common metric result assertions."""

    def _assert_results(
        results, expected_count, expected_types=None, perfect_correlation=False
    ):
        assert isinstance(results, list) and all(
            isinstance(r, MetricResult) for r in results
        )
        assert len(results) == expected_count

        if perfect_correlation:
            assert all(np.isclose(r.value, 1.0) for r in results)

        if expected_types:
            metric_types = {result.metric_type for result in results}
            assert expected_types.issubset(metric_types)

    return _assert_results
