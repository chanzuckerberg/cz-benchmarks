import pytest
import numpy as np
import anndata as ad
import pandas as pd

from czbenchmarks.datasets.utils_single_cell import (
    create_adata_for_condition,
)


@pytest.fixture
def make_adata():
    # Unified dataset: 16 cells total
    # A: 4 cells (0-3), A_small: 2 cells (4-5), B: 4 cells (6-9), NC: 6 cells (10-15)
    num_genes = 5
    var_names = [f"gene_{i}" for i in range(num_genes)]
    obs_names = [f"cell_{i:03d}" for i in range(16)]
    conditions = ["A"] * 4 + ["A_small"] * 2 + ["B"] * 4 + ["NC"] * 6

    X = np.ones((16, num_genes), dtype=float)
    # Signals
    X[0:4, 0] = 5.0  # A cells, gene_0 higher
    X[6:10, 1] = 3.0  # B cells, gene_1 higher

    adata_obj = ad.AnnData(X=X)
    adata_obj.var_names = var_names
    adata_obj.obs_names = obs_names
    adata_obj.obs["condition"] = pd.Categorical(conditions)

    # Gene-specific expressions for filters
    adata_obj.X[10:16, 4] = 0.0  # gene_4 zero in controls (NC)
    adata_obj.X[:, 2] = 0.0
    adata_obj.X[0, 2] = 1.0  # one A cell expresses gene_2
    adata_obj.X[10, 2] = 1.0  # one control cell expresses gene_2

    # Controls mapping for each condition
    control_map = {
        "A": ["cell_010", "cell_011", "cell_012"],
        "A_small": ["cell_011", "cell_012", "cell_013"],
        "B": ["cell_012", "cell_013", "cell_014"],
    }
    return adata_obj, control_map


class TestCreateAdataForCondition:
    @pytest.mark.parametrize(
        "condition,signal_gene_idx,signal_value",
        [("A", 0, 5.0), ("B", 1, 3.0)],
    )
    def test_shapes_and_labels(
        self, make_adata, condition, signal_gene_idx, signal_value
    ):
        adata_obj, control_map = make_adata

        # condition = "A"
        condition_key = "condition"
        control_name = "NC"

        rows_cond = np.where(adata_obj.obs[condition_key] == condition)[0]
        rows_ctrl = adata_obj.obs.index.get_indexer_for(control_map[condition])

        merged, num_condition = create_adata_for_condition(
            adata=adata_obj,
            condition=condition,
            condition_key=condition_key,
            control_name=control_name,
            rows_cond=rows_cond,
            rows_ctrl=rows_ctrl,
        )

        assert isinstance(merged, ad.AnnData)
        assert num_condition == len(rows_cond)
        assert merged.n_obs == len(rows_cond) + len(rows_ctrl)
        assert merged.n_vars == adata_obj.n_vars

        labels = merged.obs[condition_key].tolist()
        assert labels == [condition] * len(rows_cond) + [
            f"{control_name}_{condition}"
        ] * len(rows_ctrl)

        assert np.allclose(merged.X[: len(rows_cond), signal_gene_idx], signal_value)
        assert np.allclose(merged.X[len(rows_cond) :, signal_gene_idx], 1.0)

    def test_logs_warning_on_length_mismatch(self, make_adata, caplog):
        adata_obj, control_map = make_adata

        condition = "A_small"  # 2 condition cells vs 3 controls
        condition_key = "condition"
        control_name = "NC"
        rows_cond = np.where(adata_obj.obs[condition_key] == condition)[0]
        rows_ctrl = adata_obj.obs.index.get_indexer_for(control_map[condition])

        with caplog.at_level("WARNING"):
            _ = create_adata_for_condition(
                adata=adata_obj,
                condition=condition,
                condition_key=condition_key,
                control_name=control_name,
                rows_cond=rows_cond,
                rows_ctrl=rows_ctrl,
            )

        assert any("different lengths" in rec.message for rec in caplog.records)


# FIXME MICHELLE add test for run_multicondition_dge_analysis
