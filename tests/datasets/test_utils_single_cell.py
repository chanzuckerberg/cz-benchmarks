import pytest
import numpy as np
import anndata as ad
import pandas as pd

from czbenchmarks.datasets.utils_single_cell import run_multicondition_dge_analysis


class TestRunMulticonditionDGEAnalysis:
    @pytest.fixture
    def make_adata(self):
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

    def test_basic_returns_df_and_merged_adata(self, make_adata):
        adata_obj, control_cells_ids = make_adata

        results, merged_adata = run_multicondition_dge_analysis(
            adata=adata_obj,
            condition_key="condition",
            control_cells_ids=control_cells_ids,
            deg_test_name="wilcoxon",
            filter_min_cells=1,
            filter_min_genes=1,
            min_pert_cells=1,
            remove_avg_zeros=False,
            return_merged_adata=True,
        )

        assert isinstance(results, pd.DataFrame)
        assert results.shape[0] > 0
        assert "gene_id" in results.columns
        assert "condition" in results.columns
        # Should contain both conditions
        assert set(results["condition"]).issuperset({"A", "B"})

        # Merged AnnData should be returned and contain comparison_group labels
        assert isinstance(merged_adata, ad.AnnData)
        assert "dge_results" in merged_adata.uns
        assert "params" in merged_adata.uns["dge_results"]

        deg_results = merged_adata.uns["dge_results"]["params"]
        assert deg_results["method"] == "wilcoxon"
        assert deg_results["filter_min_cells"] == 1
        assert deg_results["filter_min_genes"] == 1
        assert deg_results["min_pert_cells"] == 1
        assert deg_results["remove_avg_zeros"] == False

        assert "comparison_group" in merged_adata.obs.columns
        groups = set(merged_adata.obs["comparison_group"].unique().tolist())
        assert {"control", "A", "B"}.issubset(groups)

    def test_returns_none_when_all_filtered_out(self, make_adata):
        adata_obj, control_cells_ids = make_adata

        # Set very strict filtering so cells are removed, triggering early None return
        results, merged_adata = run_multicondition_dge_analysis(
            adata=adata_obj,
            condition_key="condition",
            control_cells_ids=control_cells_ids,
            filter_min_cells=1,
            filter_min_genes=100,  # higher than number of genes to filter all cells out
            min_pert_cells=1,
            return_merged_adata=False,
        )

        assert results is None
        assert merged_adata is None

    def test_deg_test_name_affects_scores(self, make_adata):
        adata_obj, control_cells_ids = make_adata

        res_w, _ = run_multicondition_dge_analysis(
            adata=adata_obj,
            condition_key="condition",
            control_cells_ids=control_cells_ids,
            deg_test_name="wilcoxon",
            filter_min_cells=1,
            filter_min_genes=1,
            min_pert_cells=1,
            remove_avg_zeros=False,
            return_merged_adata=False,
        )

        res_t, _ = run_multicondition_dge_analysis(
            adata=adata_obj,
            condition_key="condition",
            control_cells_ids=control_cells_ids,
            deg_test_name="t-test",
            filter_min_cells=1,
            filter_min_genes=1,
            min_pert_cells=1,
            remove_avg_zeros=False,
            return_merged_adata=False,
        )

        # Compare scores for any condition present in both results
        conds_both = set(res_w["condition"]).intersection(set(res_t["condition"]))
        assert len(conds_both) > 0
        cond = sorted(conds_both)[0]
        s_w = (
            res_w[res_w["condition"] == cond]
            .sort_values("score", ascending=False)["score"]
            .tolist()
        )
        s_t = (
            res_t[res_t["condition"] == cond]
            .sort_values("score", ascending=False)["score"]
            .tolist()
        )
        assert s_w != s_t

    @pytest.mark.parametrize("remove_avg_zeros", [False, True])
    @pytest.mark.parametrize("return_merged_adata", [False, True])
    @pytest.mark.parametrize("filter_min_cells", [1, 3])
    @pytest.mark.parametrize("min_pert_cells", [1, 3])
    @pytest.mark.parametrize("target_condition", ["A", "A_small"])
    def test_flags_and_filters_combined(
        self,
        make_adata,
        remove_avg_zeros,
        return_merged_adata,
        filter_min_cells,
        min_pert_cells,
        target_condition,
    ):
        adata_obj, control_cells_ids = make_adata

        res, merged = run_multicondition_dge_analysis(
            adata=adata_obj,
            condition_key="condition",
            control_cells_ids=control_cells_ids,
            deg_test_name="wilcoxon",
            filter_min_cells=filter_min_cells,
            filter_min_genes=1,
            min_pert_cells=min_pert_cells,
            remove_avg_zeros=remove_avg_zeros,
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
