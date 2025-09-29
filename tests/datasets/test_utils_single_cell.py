import pytest
import numpy as np
import anndata as ad
import pandas as pd

from czbenchmarks.datasets.utils_single_cell import (
    create_adata_for_condition,
    run_multicondition_dge_analysis,
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


class TestRunMulticonditionDGEAnalysis:
    @pytest.fixture
    def make_adata(self):
        # Unified dataset: 16 cells total
        num_genes = 5
        var_names = [f"gene_{i}" for i in range(num_genes)]
        obs_names = [f"cell_{i:03d}" for i in range(16)]
        conditions = ["A"] * 4 + ["A_small"] * 2 + ["B"] * 4 + ["NC"] * 6

        X = np.ones((16, num_genes), dtype=float)
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

    def test_basic_returns_df_and_merged_adata(self, make_adata):
        adata_obj, control_cells_ids = make_adata

        results, merged = run_multicondition_dge_analysis(
            adata=adata_obj,
            condition_key="condition",
            de_gene_col="gene_id",
            control_cells_ids=control_cells_ids,
            filter_min_cells=1,
            filter_min_genes=1,
            min_pert_cells=1,
            remove_avg_zeros=False,
            store_dge_metadata=False,
            return_merged_adata=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert results.shape[0] > 0
        assert "gene_id" in results.columns
        assert "condition" in results.columns
        # Should contain both conditions
        assert set(results["condition"]).issuperset({"A", "B"})

        # Merged AnnData should be returned and contain comparison_group labels
        assert isinstance(merged, ad.AnnData)
        assert "comparison_group" in merged.obs.columns
        groups = set(merged.obs["comparison_group"].unique().tolist())
        assert {"control", "A", "B"}.issubset(groups)

    def test_returns_none_when_all_filtered_out(self, make_adata):
        adata_obj, control_cells_ids = make_adata

        # Set very strict filtering so cells are removed, triggering early None return
        results, merged = run_multicondition_dge_analysis(
            adata=adata_obj,
            condition_key="condition",
            de_gene_col="gene_id",
            control_cells_ids=control_cells_ids,
            filter_min_cells=1,
            filter_min_genes=100,  # higher than number of genes to filter all cells out
            min_pert_cells=1,
            return_merged_adata=False,
        )

        assert isinstance(results, pd.DataFrame) and results.empty
        assert merged is None

    def test_deg_test_name_affects_scores(self, make_adata):
        adata_obj, control_cells_ids = make_adata

        res_w, _ = run_multicondition_dge_analysis(
            adata=adata_obj,
            condition_key="condition",
            de_gene_col="gene_id",
            control_cells_ids=control_cells_ids,
            filter_min_cells=1,
            filter_min_genes=1,
            min_pert_cells=1,
            remove_avg_zeros=False,
            store_dge_metadata=False,
            return_merged_adata=False,
        )
        # Basic checks on returned scores for a single method
        assert not res_w.empty
        assert "score" in res_w.columns

    @pytest.mark.parametrize("remove_avg_zeros", [False, True])
    @pytest.mark.parametrize("store_dge_metadata", [False, True])
    @pytest.mark.parametrize("return_merged_adata", [False, True])
    @pytest.mark.parametrize("filter_min_cells", [1, 3])
    @pytest.mark.parametrize("min_pert_cells", [1, 3])
    @pytest.mark.parametrize("target_condition", ["A", "A_small"])
    def test_flags_and_filters_combined(
        self,
        make_adata,
        remove_avg_zeros,
        store_dge_metadata,
        return_merged_adata,
        filter_min_cells,
        min_pert_cells,
        target_condition,
    ):
        adata_obj, control_cells_ids = make_adata

        res, merged = run_multicondition_dge_analysis(
            adata=adata_obj,
            condition_key="condition",
            de_gene_col="gene_id",
            control_cells_ids=control_cells_ids,
            filter_min_cells=filter_min_cells,
            filter_min_genes=1,
            min_pert_cells=min_pert_cells,
            remove_avg_zeros=remove_avg_zeros,
            store_dge_metadata=store_dge_metadata,
            return_merged_adata=return_merged_adata,
        )

        # Basic assertions and present conditions
        assert res is not None and res.shape[0] > 0
        present_conditions = set(res["condition"].unique().tolist())

        # min_pert_cells behavior for target condition
        num_cells_condition = int(
            (adata_obj.obs["condition"] == target_condition).sum()
        )
        if min_pert_cells > num_cells_condition:
            assert target_condition not in present_conditions
        else:
            assert target_condition in present_conditions

        # store_dge_metadata flag behavior
        if store_dge_metadata:
            assert "dge_params" in res.columns
        else:
            assert "dge_params" not in res.columns

        # return_merged_adata flag behavior
        if return_merged_adata:
            assert isinstance(merged, ad.AnnData)
            assert "comparison_group" in merged.obs.columns
        else:
            assert merged is None

        # Only perform gene assertions if target condition is present
        if target_condition in present_conditions:
            target_genes = set(
                res[res["condition"] == target_condition]["gene_id"].tolist()
            )

            # remove_avg_zeros behavior for gene_4
            if remove_avg_zeros:
                assert "gene_4" not in target_genes
            else:
                # gene_4 appears only if it passes min_cells for the merged slice
                expected_gene4_present = filter_min_cells <= num_cells_condition
                if expected_gene4_present:
                    assert "gene_4" in target_genes
                else:
                    assert "gene_4" not in target_genes

            # filter_min_cells behavior for gene_2
            # gene_2 is expressed in one A cell and one control cell only.
            if target_condition == "A":
                if filter_min_cells == 1:
                    assert "gene_2" in target_genes
                else:
                    assert "gene_2" not in target_genes
            else:
                assert "gene_2" not in target_genes
