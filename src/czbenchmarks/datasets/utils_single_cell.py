from typing import Dict, List, Literal, Tuple
import logging

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm


logger = logging.getLogger(__name__)


def create_adata_for_condition(
    adata: ad.AnnData,
    condition: str,
    condition_key: str,
    control_name: str,
    rows_cond: np.ndarray[np.int_],
    rows_ctrl: np.ndarray[np.int_],
) -> Tuple[ad.AnnData, int]:
    """
    Create an AnnData object for a single condition.

    Args:
        adata: ad.AnnData, anndata with condition and control cells
        condition: str, condition to create paired condition / control adata
        condition_key: str, condition key in adata.obs
        control_name: str, control name in adata.obs
        rows_cond: ``np.ndarray[np.int_]``, integer rows of condition cells in adata
        rows_ctrl: ``np.ndarray[np.int_]``, integer rows of matched control cells in adata

    Returns:
        adata_merged: ad.AnnData, adata with condition and control cells
        num_condition: int, number of condition cells
    """

    adata_condition = adata[rows_cond]
    adata_control = adata[rows_ctrl]

    if len(adata_condition) != len(adata_control):
        logger.warning(
            f"Condition and control data for {condition} have different lengths."
        )

    # Concatenate condition and control data
    adata_merged = ad.concat([adata_condition, adata_control], index_unique=None).copy()

    label_cond = [condition] * len(adata_condition)
    label_ctrl = [f"{control_name}_{condition}"] * len(adata_control)
    adata_merged.obs[condition_key] = label_cond + label_ctrl

    # Add condition to cell_barcode_gene column and set as index
    adata_merged.obs_names = adata_merged.obs_names.astype(str) + "_" + condition

    return adata_merged, len(adata_condition)


def run_multicondition_dge_analysis(
    adata: ad.AnnData,
    condition_key: str,
    control_name: str,
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
    control_name (str): Name of the control condition.
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
        logger.warning(
            "return_merged_adata is True, which can consume a large amount of memory."
        )

    obs = adata.obs
    obs_index = obs.index

    # Ensure categorical dtype for faster grouping
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
    with tqdm(
        total=len(target_conditions), desc="Processing de conditions", unit="item"
    ) as pbar:
        for selected_condition in target_conditions:
            # Skip conditions with insufficient perturbed cells
            if len(condition_to_indices[selected_condition]) < min_pert_cells:
                pbar.set_postfix_str(f"Skipped {selected_condition}: min_pert_cells")
                pbar.update(1)
                continue
            adata_merged, num_condition = create_adata_for_condition(
                adata=adata,
                condition=selected_condition,
                condition_key=condition_key,
                control_name=control_name,
                rows_cond=condition_to_indices[selected_condition],
                rows_ctrl=control_to_indices[selected_condition],
            )

            # Add simple comparison group label for rank_genes_groups
            comparison_group_col = "comparison_group"
            adata_merged.obs[comparison_group_col] = selected_condition
            control_idx = np.arange(num_condition, adata_merged.n_obs)
            adata_merged.obs.iloc[
                control_idx, adata_merged.obs.columns.get_loc(comparison_group_col)
            ] = control_name

            # Normalize and filter
            if deg_test_name == "wilcoxon":
                logger.info(
                    "Normalizing total counts and log transforming for Wilcoxon test"
                )
                sc.pp.normalize_total(adata_merged, target_sum=1e4)
                sc.pp.log1p(adata_merged)
            elif deg_test_name == "t-test":
                logger.info("Calculating z-scores for genes by gem group for T-test")
                logger.warning("This is not implemented yet")
                # FIXME MICHELLE: calculate z-scores for genes by gem group
            n_vars = adata_merged.n_vars
            sc.pp.filter_genes(adata_merged, min_cells=filter_min_cells)
            if n_vars != adata_merged.n_vars:
                logger.info(
                    f"Filtered {n_vars - adata_merged.n_vars} genes using min_cells={filter_min_cells}"
                )

            n_obs = adata_merged.n_obs
            sc.pp.filter_cells(adata_merged, min_genes=filter_min_genes)
            if n_obs != adata_merged.n_obs:
                logger.info(
                    f"Filtered {n_obs - adata_merged.n_obs} cells using min_genes={filter_min_genes}"
                )

            comparison_group_counts = adata_merged.obs[
                comparison_group_col
            ].value_counts()
            if len(comparison_group_counts) < 2 or comparison_group_counts.min() < 1:
                logger.warning(
                    f"Insufficient filtered cells for analysis of {selected_condition}"
                )
                continue

            # Run statistical test
            sc.tl.rank_genes_groups(
                adata_merged,
                groupby=comparison_group_col,
                reference=control_name,
                method=deg_test_name,
                key_added="dge_results",
            )

            # Get results DataFrame
            results = sc.get.rank_genes_groups_df(
                adata_merged, group=selected_condition, key="dge_results"
            )
            results[condition_key] = selected_condition
            if deg_test_name == "t-test":
                n = len(adata_merged)
                effective_n = np.sqrt(4 / n)
                results["standardized_mean_diff"] = results["scores"] * effective_n

            # Option to remove zero expression genes
            if remove_avg_zeros:
                # Filtering can change cells so need to recalculate masks
                comp_vals = adata_merged.obs[comparison_group_col].to_numpy()
                control_mask = comp_vals == control_name
                condition_mask = comp_vals == selected_condition
                nc_mean = (
                    adata_merged[control_mask, results["names"]]
                    .X.mean(axis=0)
                    .flatten()
                )
                target_mean = (
                    adata_merged[condition_mask, results["names"]]
                    .X.mean(axis=0)
                    .flatten()
                )
                indexes = np.where((target_mean > 0) & (nc_mean > 0))[0]
                logger.info(
                    f"remove_avg_zeros is True. Removing {len(results) - len(indexes)} genes with zero expression"
                )
                results = results.iloc[indexes]

            results_df.append(results)
            if return_merged_adata:
                adata_merged.obs.drop(columns=[comparison_group_col], inplace=True)
                adata_results.append(adata_merged)

            pbar.set_postfix_str(f"Completed {pbar.n + 1}/{len(target_conditions)}")
            pbar.update(1)

    if len(results_df) > 0:
        results_df = pd.concat(results_df, ignore_index=True)

        # Standardize column names
        col_mapper = {
            "names": "gene_id",
            "group": "group",
            "scores": "score",
            "logfoldchanges": "logfoldchange",
            "pvals": "pval",
            "pvals_adj": "pval_adj",
            "standardized_mean_diff": "standardized_mean_diff",
        }
        results_df = results_df.rename(columns=col_mapper)
        cols = [condition_key] + [
            x for x in col_mapper.values() if x in results_df.columns
        ]
        results_df = results_df[cols]
    else:
        logger.error(
            "No differential expression results were produced. All conditions were skipped after filtering. "
            f"Parameters: filter_min_cells={filter_min_cells}, filter_min_genes={filter_min_genes}, min_pert_cells={min_pert_cells}"
        )
        raise ValueError(
            "No differential expression results were produced. Try relaxing filtering thresholds or checking inputs."
        )

    # Create merged anndata if it will be returned
    if return_merged_adata:
        dge_params = adata_results[0].uns["dge_results"]["params"].copy()
        dge_params.update(
            {
                "remove_avg_zeros": remove_avg_zeros,
                "filter_min_cells": filter_min_cells,
                "filter_min_genes": filter_min_genes,
                "min_pert_cells": min_pert_cells,
            }
        )

        adata_merged = ad.concat(adata_results, index_unique=None)
        del adata_results
        adata_merged.uns["dge_results"] = {"params": dge_params}
        return results_df, adata_merged

    return results_df
