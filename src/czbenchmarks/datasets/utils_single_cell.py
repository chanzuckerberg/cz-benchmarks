from typing import Tuple, Dict, List
import logging
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from pandas.api.types import is_categorical_dtype
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def run_multicondition_dge_analysis(
    adata: ad.AnnData,
    condition_key: str,
    de_gene_col: str,
    control_cells_ids: Dict[str, List[str]],
    filter_min_cells: int = 10,
    filter_min_genes: int = 1000,
    min_pert_cells: int = 50,
    remove_avg_zeros: bool = False,
    store_dge_metadata: bool = False,
    return_merged_adata: bool = False,
) -> Tuple[pd.DataFrame, ad.AnnData]:
    """
    Run differential gene expression analysis for a list of conditions between perturbed
        and matched control cells.

    Parameters
    ----------
    adata (AnnData): Annotated data matrix containing gene expression and metadata.
    condition_key (str): Column name for condition labels in `adata.obs`.
    de_gene_col (str): Column name for gene labels in `adata.var`.
    control_cells_ids (Dict[str, List[str]]): Mapping from condition -> list of matched control cell ids.
    filter_min_cells (int, optional): Minimum number of cells expressing a gene to include that gene. Defaults to 10.
    filter_min_genes (int, optional): Minimum number of genes detected per cell. Defaults to 1000.
    min_pert_cells (int, optional): Minimum number of perturbed cells required. Defaults to 50.
    remove_avg_zeros (bool, optional): Whether to remove genes with zero average expression. Defaults to True.
    store_dge_metadata (bool, optional): Whether to store DGE metadata in the results DataFrame. Defaults to False.
    return_merged_adata (bool, optional): Whether to return the merged AnnData object. Defaults to False.

    Returns
    -------
    Tuple[pd.DataFrame, anndata.AnnData]
        (results_df, adata_merged):
        - results_df: Differential expression results for `selected_condition`.
        - adata_merged: AnnData containing concatenated condition and control cells.
    """

    if return_merged_adata:
        logger.warning(
            "return_merged_adata is True, which can consume a large amount of memory."
        )

    deg_test_name = "wilcoxon"
    obs = adata.obs
    obs_index = obs.index

    # Optional: ensure categorical for faster grouping
    if not is_categorical_dtype(obs[condition_key].dtype):
        obs[condition_key] = obs[condition_key].astype("category")

    # condition -> integer row positions
    condition_to_indices = obs.groupby(condition_key, observed=True).indices

    # control ids -> integer row positions per condition (preserves order)
    control_to_indices = {
        cond: obs_index.get_indexer_for(ids) for cond, ids in control_cells_ids.items()
    }

    target_conditions = list(control_cells_ids.keys())
    adata_results = []
    results_df = []

    # Condition loop starts here
    for selected_condition in tqdm(
        target_conditions, desc="Analyzing conditions", unit="cond"
    ):
        rows_cond = condition_to_indices.get(
            selected_condition, np.array([], dtype=int)
        )
        rows_ctrl = control_to_indices.get(selected_condition, np.array([], dtype=int))

        # Filter out any missing indices (-1)
        rows_ctrl = np.asarray(rows_ctrl, dtype=int)
        rows_ctrl = rows_ctrl[rows_ctrl >= 0]

        if len(rows_cond) < min_pert_cells or len(rows_ctrl) == 0:
            print(f"Insufficient cells for analysis of {selected_condition}")
            continue

        # Create condition and control data, then concatenate
        # Copy slices to avoid ImplicitModificationWarning when editing .obs
        adata_condition = adata[rows_cond].copy()
        adata_control = adata[rows_ctrl].copy()

        if len(adata_condition) != len(adata_control):
            logger.warning(
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
        sc.pp.filter_cells(adata_merged, min_genes=filter_min_genes)
        sc.pp.filter_genes(adata_merged, min_cells=filter_min_cells)
        sc.pp.normalize_total(adata_merged, target_sum=1e4)
        sc.pp.log1p(adata_merged)

        comparison_group_counts = adata_merged.obs["comparison_group"].value_counts()
        if len(comparison_group_counts) < 2 or comparison_group_counts.min() < 1:
            logger.warning(
                f"Insufficient filtered cells for analysis of {selected_condition}"
            )
            continue

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
        results[condition_key] = selected_condition

        # Option to remove zero expression genes
        if remove_avg_zeros:
            gene_ids = results["names"].values
            cond_view = adata_condition[:, gene_ids]
            ctrl_view = adata_control[:, gene_ids]
            cond_mean = cond_view.X.mean(axis=0)
            ctrl_mean = ctrl_view.X.mean(axis=0)
            # handle sparse vs dense
            cond_mean = (
                cond_mean.A1
                if hasattr(cond_mean, "A1")
                else np.asarray(cond_mean).ravel()
            )
            ctrl_mean = (
                ctrl_mean.A1
                if hasattr(ctrl_mean, "A1")
                else np.asarray(ctrl_mean).ravel()
            )
            indexes = np.where((cond_mean > 0) & (ctrl_mean > 0))[0]
            logger.info(
                f"remove_avg_zeros is True. Removing {len(results) - len(indexes)} genes with zero expression"
            )
            results = results.iloc[indexes]

        results_df.append(results)
        if return_merged_adata:
            adata_results.append(adata_merged)

    if not results_df:
        return pd.DataFrame(), None

    results = pd.concat(results_df, ignore_index=True)
    del results_df

    # dge params captured from last run
    try:
        dge_params = adata_merged.uns["dge_results"]["params"]  # type: ignore[name-defined]
    except Exception:
        dge_params = {}

    if return_merged_adata:
        adata_merged = ad.concat(adata_results, index_unique=None)
        del adata_results
    else:
        adata_merged = None

    # Standardize column names
    col_mapper = {
        "names": de_gene_col,
        "scores": "score",
        "logfoldchanges": "logfoldchange",
        "pvals": "pval",
        "pvals_adj": "pval_adj",
    }
    results = results.rename(columns=col_mapper)
    cols = [x for x in col_mapper.values() if x in results.columns] + [condition_key]
    results = results[cols]

    if store_dge_metadata:
        dge_params.update(
            {
                "remove_avg_zeros": remove_avg_zeros,
                "filter_min_cells": filter_min_cells,
                "filter_min_genes": filter_min_genes,
                "min_pert_cells": min_pert_cells,
            }
        )
        results["dge_params"] = dge_params  # NB: this is not tidy
    return results, adata_merged
