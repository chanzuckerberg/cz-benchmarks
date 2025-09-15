from typing import Dict, List, Literal, Tuple
import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


log = logging.getLogger(__name__)


def run_multicondition_dge_analysis(
    adata: ad.AnnData,
    condition_key: str,
    control_cells_ids: Dict[str, List[str]],
    deg_test_name: Literal["wilcoxon", "t-test"] = "wilcoxon",
    filter_min_cells: int = 10,
    filter_min_genes: int = 1000,
    min_pert_cells: int = 50,
    remove_avg_zeros: bool = False,
    return_merged_adata: bool = False,
) -> Tuple[pd.DataFrame, ad.AnnData]:
    """
    Run differential gene expression analysis for a list of conditions between perturbed
        and matched control cells.

    Parameters
    ----------
    adata (AnnData): Annotated data matrix containing gene expression and metadata.
    condition_key (str): Column name for condition labels in `adata.obs`.
    control_cells_ids (Dict[str, List[str]]): Mapping from condition -> list of matched control cell ids.
    deg_test_name (Literal["wilcoxon", "t-test"], optional): Statistical test name for differential expression. Defaults to 'wilcoxon'.
    filter_min_cells (int, optional): Minimum number of cells expressing a gene to include that gene. Defaults to 10.
    filter_min_genes (int, optional): Minimum number of genes detected per cell. Defaults to 1000.
    min_pert_cells (int, optional): Minimum number of perturbed cells required. Defaults to 50.
    remove_avg_zeros (bool, optional): Whether to remove genes with zero average expression. Defaults to True.
    return_merged_adata (bool, optional): Whether to return the merged AnnData object. Defaults to False.

    Returns
    -------
    Tuple[pd.DataFrame, anndata.AnnData]
        (results_df, adata_merged):
        - results_df: Differential expression results for `selected_condition`.
        - adata_merged: AnnData containing concatenated condition and control cells.
    """

    if deg_test_name not in ["wilcoxon", "t-test"]:
        raise ValueError(
            f"Invalid deg_test_name: {deg_test_name}. Must be 'wilcoxon' or 't-test'."
        )

    if return_merged_adata:
        log.warning(
            "return_merged_adata is True, which can consume a large amount of memory."
        )

    obs = adata.obs
    obs_index = obs.index

    # Optional: ensure categorical for faster grouping
    if not isinstance(obs[condition_key], pd.CategoricalDtype):
        obs[condition_key] = pd.Categorical(obs[condition_key])

    # condition -> integer row positions
    condition_to_indices = obs.groupby(condition_key, observed=True).indices

    # control ids -> integer row positions per condition (preserves order)
    control_to_indices = {
        cond: obs_index.get_indexer_for(ids) for cond, ids in control_cells_ids.items()
    }

    target_conditions = control_cells_ids.keys()
    adata_results = []
    results_df = []

    # Condition loop starts here
    for selected_condition in target_conditions:
        rows_cond = condition_to_indices.get(
            selected_condition, np.array([], dtype=int)
        )
        rows_ctrl = control_to_indices.get(selected_condition, np.array([], dtype=int))
        # Filter out any missing indices (-1)
        rows_ctrl = (
            rows_ctrl[rows_ctrl >= 0]
            if isinstance(rows_ctrl, np.ndarray)
            else np.array(rows_ctrl, dtype=int)
        )

        if len(rows_cond) < min_pert_cells or len(rows_ctrl) == 0:
            log.warning(f"Insufficient cells for analysis of {selected_condition}")
            continue

        # Create condition and control data, then concatenate
        adata_condition = adata[rows_cond]
        adata_control = adata[rows_ctrl]

        if len(adata_condition) != len(adata_control):
            log.warning(
                f"Condition and control data for {selected_condition} have different lengths."
            )

        if adata.isbacked:
            adata_condition = adata_condition.to_memory()
            adata_control = adata_control.to_memory()

        # Add comparison group label to each slice before concatenation
        adata_condition.obs["comparison_group"] = selected_condition
        adata_control.obs["comparison_group"] = "control"
        adata_merged = ad.concat(
            [adata_condition, adata_control], index_unique=None
        ).copy()

        # Normalize and filter
        sc.pp.normalize_total(adata_merged, target_sum=1e4)
        sc.pp.log1p(adata_merged)
        sc.pp.filter_genes(adata_merged, min_cells=filter_min_cells)
        sc.pp.filter_cells(adata_merged, min_genes=filter_min_genes)

        comparison_group_counts = adata_merged.obs["comparison_group"].value_counts()
        if len(comparison_group_counts) < 2 or comparison_group_counts.min() < 1:
            log.warning(
                f"Insufficient filtered cells for analysis of {selected_condition}"
            )
            return None, None

        # Run statistical test
        sc.tl.rank_genes_groups(
            adata_merged,
            groupby="comparison_group",
            reference="control",
            method=deg_test_name,
            key_added="dge_results",
        )

        # Get results DataFrame
        results = sc.get.rank_genes_groups_df(
            adata_merged, group=selected_condition, key="dge_results"
        )
        # Add condition name
        results["condition"] = selected_condition

        # Option to remove zero expression genes
        if remove_avg_zeros:
            target_mean = adata_condition[:, results.names].X.mean(axis=0).flatten()
            nc_mean = adata_control[:, results.names].X.mean(axis=0).flatten()
            indexes = np.where((target_mean > 0) & (nc_mean > 0))[0]
            log.info(
                f"remove_avg_zeros is True. Removing {len(results) - len(indexes)} genes with zero expression"
            )
            results = results.iloc[indexes]

        results_df.append(results)
        if return_merged_adata:
            adata_results.append(adata_merged)

    results_df = pd.concat(results_df, ignore_index=True)

    if return_merged_adata:
        dge_params = adata_results[0].uns["dge_results"]["params"].copy()
        adata_merged = ad.concat(adata_results, index_unique=None)
        del adata_results
        dge_params.update(
            {
                "remove_avg_zeros": remove_avg_zeros,
                "filter_min_cells": filter_min_cells,
                "filter_min_genes": filter_min_genes,
                "min_pert_cells": min_pert_cells,
            }
        )
        adata_merged.uns["dge_results"] = {"params": dge_params}
    else:
        adata_merged = None

    # Standardize column names
    col_mapper = {
        "names": "gene_id",
        "scores": "score",
        "logfoldchanges": "logfoldchange",
        "pvals": "pval",
        "pvals_adj": "pval_adj",
        "smd": "standardized_mean_diff",
        "group": "group",
        "condition": "condition",
    }
    results_df = results_df.rename(columns=col_mapper)
    cols = [x for x in col_mapper.values() if x in results_df.columns]
    results_df = results_df[cols]
    return results_df, adata_merged
