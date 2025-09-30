import numpy as np
import pandas as pd
import anndata as ad
import pytest


def test_pipeline_stores_control_mapping_and_de_results(monkeypatch, tmp_path):
    from dge_files import preprocess_pipeline as pp

    # Build a tiny AnnData with 2 controls and 2 treated cells for one condition
    X = np.array(
        [
            [1.0, 2.0],  # NT1 (control)
            [2.0, 1.0],  # NT2 (control)
            [2.0, 3.0],  # T1 (treated)
            [3.0, 2.0],  # T2 (treated)
        ]
    )
    obs = pd.DataFrame(
        {
            "condition": ["non-targeting", "non-targeting", "pertA", "pertA"],
            "gem_group": ["A", "A", "A", "A"],
        },
        index=["NT1", "NT2", "T1", "T2"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Stub matched controls to produce a strict 1-1 mapping
    def _stub_get_matched_controls(**kwargs):
        return {"T1": "NT1", "T2": "NT2"}

    # Patch the symbol used by the pipeline module
    monkeypatch.setattr(
        "dge_files.preprocess_pipeline.get_matched_controls",
        _stub_get_matched_controls,
    )

    # Stub DE to return a small DataFrame
    def _stub_run_multicondition_dge_analysis(**kwargs):
        df = pd.DataFrame(
            {
                "condition": ["pertA", "pertA"],
                "gene_id": ["g1", "g2"],
                "logfoldchange": [1.0, -1.0],
                "pval_adj": [1e-6, 1e-3],
            }
        )
        return df, None

    # Patch the symbol used by the pipeline module
    monkeypatch.setattr(
        "dge_files.preprocess_pipeline.run_multicondition_dge_analysis",
        _stub_run_multicondition_dge_analysis,
    )

    # Write input h5ad
    in_path = tmp_path / "toy.h5ad"
    adata.write_h5ad(in_path)

    # Run
    out_adata = pp.run_pipeline(
        str(in_path),
        condition_col="condition",
        control_name="non-targeting",
        filter_min_cells=0,
        filter_min_genes=0,
        min_pert_cells=1,
    )

    # Assertions: control mapping and ids
    assert "control_cells_map" in out_adata.uns
    assert out_adata.uns["control_cells_map"]["pertA"] == {"T1": "NT1", "T2": "NT2"}
    assert "control_cells_ids" in out_adata.uns
    assert set(out_adata.uns["control_cells_ids"]["pertA"]) == {"NT1", "NT2"}

    # Assertions: DE results
    assert "de_results_wilcoxon" in out_adata.uns
    de = out_adata.uns["de_results_wilcoxon"]
    assert isinstance(de, pd.DataFrame)
    assert set(["condition", "gene", "logfoldchange", "pval_adj"]).issuperset(
        set(de.columns)
    )
    assert (de["condition"] == "pertA").all()


def test_control_matching_respects_gem_group_and_min_cells(monkeypatch, tmp_path):
    from dge_files import preprocess_pipeline as pp

    # Build AnnData with two gem groups and two perturbation conditions
    # Controls: A-group (NTA1, NTA2), B-group (NTB1)
    # Treated: pertA in A-group (T1A, T2A), pertB in B-group (T1B)
    X = np.array(
        [
            [1.0, 2.0],  # NTA1
            [2.0, 1.0],  # NTA2
            [1.5, 1.5],  # NTB1
            [3.0, 4.0],  # T1A
            [4.0, 3.0],  # T2A
            [1.2, 1.1],  # T1B
        ]
    )
    obs = pd.DataFrame(
        {
            "condition": [
                "non-targeting",
                "non-targeting",
                "non-targeting",
                "pertA",
                "pertA",
                "pertB",
            ],
            "gem_group": ["A", "A", "B", "A", "A", "B"],
        },
        index=["NTA1", "NTA2", "NTB1", "T1A", "T2A", "T1B"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    def _smart_match(adata, perturbation, min_cells, verbose, dict_ctrls, pert_column, ctrl_condition, gem_column, libsize_column, ngenes_column):
        # Enforce threshold on treated cells
        treated = adata.obs.index[adata.obs[pert_column] == perturbation].tolist()
        if len(treated) < int(min_cells):
            return {}
        # Match within the same gem group, 1-1 in order
        mapping = {}
        for t in treated:
            tg = adata.obs.loc[t, gem_column]
            ctrls = adata.obs.index[(adata.obs[pert_column] == ctrl_condition) & (adata.obs[gem_column] == tg)].tolist()
            if len(ctrls) == 0:
                continue
            # Deterministic: pick the first available control in that group
            mapping[t] = ctrls[0] if t.endswith("A") else ctrls[-1]
        return mapping

    monkeypatch.setattr(
        "dge_files.preprocess_pipeline.get_matched_controls",
        _smart_match,
    )

    # Capture filtering args passed into DE runner and return dummy DE
    captured = {}

    def _stub_run_multicondition_dge_analysis(**kwargs):
        captured["filter_min_cells"] = kwargs.get("filter_min_cells")
        captured["filter_min_genes"] = kwargs.get("filter_min_genes")
        captured["min_pert_cells"] = kwargs.get("min_pert_cells")
        df = pd.DataFrame(
            {
                "condition": ["pertA", "pertB"],
                "gene_id": ["g1", "g2"],
                "logfoldchange": [1.0, 0.5],
                "pval_adj": [1e-5, 1e-2],
            }
        )
        return df, None

    monkeypatch.setattr(
        "dge_files.preprocess_pipeline.run_multicondition_dge_analysis",
        _stub_run_multicondition_dge_analysis,
    )

    in_path = tmp_path / "toy2.h5ad"
    adata.write_h5ad(in_path)

    out_adata = pp.run_pipeline(
        str(in_path),
        condition_col="condition",
        control_name="non-targeting",
        filter_min_cells=10,
        filter_min_genes=1000,
        min_pert_cells=1,
    )

    # Control mapping respects gem_group: pertA maps to A controls only; pertB maps to B controls only
    cmap = out_adata.uns["control_cells_map"]
    assert set(cmap["pertA"].values()) <= {"NTA1", "NTA2"}
    assert set(cmap["pertA"].keys()) == {"T1A", "T2A"}
    assert set(cmap["pertB"].values()) == {"NTB1"}
    assert set(cmap["pertB"].keys()) == {"T1B"}

    # DE runner received filtering arguments we passed
    assert captured == {"filter_min_cells": 10, "filter_min_genes": 1000, "min_pert_cells": 1}

    # Raise min_pert_cells above pertB treated count (1) but not above pertA (2)
    out_adata2 = pp.run_pipeline(
        str(in_path),
        condition_col="condition",
        control_name="non-targeting",
        filter_min_cells=0,
        filter_min_genes=0,
        min_pert_cells=2,
    )
    cmap2 = out_adata2.uns["control_cells_map"]
    assert "pertA" in cmap2 and len(cmap2["pertA"]) > 0
    # pertB should be absent or empty due to threshold
    assert "pertB" not in cmap2 or len(cmap2["pertB"]) == 0


def test_de_empty_when_no_matches(monkeypatch, tmp_path):
    from dge_files import preprocess_pipeline as pp

    X = np.array([[1.0, 1.0], [1.1, 0.9], [2.0, 2.0]])
    obs = pd.DataFrame(
        {"condition": ["non-targeting", "pertA", "pertA"], "gem_group": ["A", "A", "A"]},
        index=["NT1", "T1", "T2"],
    )
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # No matches returned
    def _no_match(**kwargs):
        return {}

    monkeypatch.setattr(
        "dge_files.preprocess_pipeline.get_matched_controls",
        _no_match,
    )

    # DE is not even called with empty mapping; but stub anyway if called
    monkeypatch.setattr(
        "dge_files.preprocess_pipeline.run_multicondition_dge_analysis",
        lambda **kwargs: (pd.DataFrame(), None),
    )

    in_path = tmp_path / "toy3.h5ad"
    adata.write_h5ad(in_path)

    out_adata = pp.run_pipeline(
        str(in_path),
        condition_col="condition",
        control_name="non-targeting",
        filter_min_cells=0,
        filter_min_genes=0,
        min_pert_cells=1,
    )
    de = out_adata.uns["de_results_wilcoxon"]
    assert isinstance(de, pd.DataFrame)
    assert de.shape[0] == 0
    assert set(["condition", "gene", "logfoldchange", "pval_adj"]).issuperset(set(de.columns))


def test_errors_missing_columns_or_control(monkeypatch, tmp_path):
    from dge_files import preprocess_pipeline as pp

    # Missing condition column
    X = np.array([[1.0, 1.0], [2.0, 2.0]])
    obs = pd.DataFrame({}, index=["C1", "C2"])  # no 'condition'
    var = pd.DataFrame(index=["g1", "g2"])
    adata = ad.AnnData(X=X, obs=obs, var=var)
    p = tmp_path / "missing_cond.h5ad"
    adata.write_h5ad(p)
    with pytest.raises(ValueError):
        pp.run_pipeline(str(p))

    # No control present
    obs2 = pd.DataFrame({"condition": ["pertA", "pertA"]}, index=["T1", "T2"])  # no 'non-targeting'
    adata2 = ad.AnnData(X=X, obs=obs2, var=var)
    p2 = tmp_path / "no_ctrl.h5ad"
    adata2.write_h5ad(p2)
    with pytest.raises(ValueError):
        pp.run_pipeline(str(p2))



def test_real_control_matching_larger_dataset(monkeypatch, tmp_path):
    from dge_files import preprocess_pipeline as pp

    # Build a larger AnnData with multiple conditions and gem groups
    # Controls: A-group (CTA1, CTA2, CTA3), B-group (CTB1)
    # Treated per condition:
    #   pertA: A-group (TA1, TA2), B-group (TB1, TB2)
    #   pertB: A-group (TB_A_only1) -> below threshold when min_pert_cells=2
    #   pertC: C-group (TC1_C, TC2_C) -> no controls available in group C
    obs = pd.DataFrame(
        {
            "condition": [
                # controls
                "non-targeting", "non-targeting", "non-targeting", "non-targeting",
                # pertA
                "pertA", "pertA", "pertA", "pertA",
                # pertB (below threshold)
                "pertB",
                # pertC (no matching controls in group C)
                "pertC", "pertC",
            ],
            "gem_group": [
                # controls
                "A", "A", "A", "B",
                # pertA
                "A", "A", "B", "B",
                # pertB
                "A",
                # pertC
                "C", "C",
            ],
        },
        index=[
            # controls
            "CTA1", "CTA2", "CTA3", "CTB1",
            # pertA
            "TA1", "TA2", "TB1", "TB2",
            # pertB
            "TB_A_only1",
            # pertC
            "TC1_C", "TC2_C",
        ],
    )
    # Create a simple expression matrix (values only affect UMI/ngenes features)
    X = np.array([
        [1.0, 2.0, 3.0],  # CTA1
        [2.0, 1.0, 3.0],  # CTA2
        [3.0, 1.0, 2.0],  # CTA3
        [1.5, 2.5, 3.5],  # CTB1
        [4.0, 1.0, 0.5],  # TA1
        [0.5, 4.0, 1.0],  # TA2
        [3.5, 2.5, 0.1],  # TB1
        [0.1, 3.5, 2.5],  # TB2
        [5.0, 0.2, 0.2],  # TB_A_only1
        [1.0, 1.0, 1.0],  # TC1_C
        [2.0, 2.0, 2.0],  # TC2_C
    ])
    var = pd.DataFrame(index=["g1", "g2", "g3"])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Stub DE to keep the test fast; use real control-matching logic
    monkeypatch.setattr(
        "dge_files.preprocess_pipeline.run_multicondition_dge_analysis",
        lambda **kwargs: (pd.DataFrame(), None),
    )

    in_path = tmp_path / "larger.h5ad"
    adata.write_h5ad(in_path)

    out_adata = pp.run_pipeline(
        str(in_path),
        condition_col="condition",
        control_name="non-targeting",
        filter_min_cells=0,
        filter_min_genes=0,
        min_pert_cells=2,  # threshold excludes pertB (only 1 treated)
    )

    assert "control_cells_map" in out_adata.uns
    cmap = out_adata.uns["control_cells_map"]

    # Only pertA should have matches; pertB is below threshold; pertC has no controls in group C
    assert "pertA" in cmap
    assert "pertB" not in cmap
    assert "pertC" not in cmap or len(cmap.get("pertC", {})) == 0

    # For pertA: group A has 2 treated, 3 controls -> 2 matches; group B has 2 treated, 1 control -> 1 match
    # Total expected matches = 3, with unique control usage and same-group pairing
    mapping = cmap["pertA"]
    assert isinstance(mapping, dict)
    assert len(mapping) == 3
    matched_ctrls = list(mapping.values())
    assert len(set(matched_ctrls)) == len(matched_ctrls)  # no control reused

    # Verify same-group pairing from obs metadata
    obs_df = out_adata.obs
    for treated_id, ctrl_id in mapping.items():
        tg = obs_df.loc[treated_id, "gem_group"]
        cg = obs_df.loc[ctrl_id, "gem_group"]
        assert tg == cg

    # Verify all treated keys belong to pertA
    treated_ids = set(obs_df.index[obs_df["condition"] == "pertA"].tolist())
    assert set(mapping.keys()).issubset(treated_ids)
