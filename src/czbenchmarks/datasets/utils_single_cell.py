from typing import Dict, List, Literal, Tuple, Optional
import logging
import os
import numba as nb
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from scipy.sparse import csr_matrix
from tqdm import tqdm

logger = logging.getLogger(__name__)

# NB: sparse_mean_var_minor_axis has been moved to a new package (fast-array-utils)
# in main branch of scanpy, but not yet released. Try/except is for when
# cz-benchmarks supports scanpy versions with different requirements, but should 
# be removed once all supported scanpy versions require fast-array-utils
try:
    from scanpy.preprocessing._utils import sparse_mean_var_minor_axis
    logger.info("Using sparse_mean_var_minor_axis from scanpy.preprocessing._utils")
except ImportError:
    from fast_array_utils import sparse_mean_var_minor_axis
    logger.info("Using sparse_mean_var_minor_axis from fast_array_utils")

CPU_COUNT = os.cpu_count()

# FIXME MICHELLE remove this for the DGE free version of code
@nb.njit(parallel=True, fastmath=True)
def colwise_nonzero_mean_var_numba(X):
    n_rows, n_cols = X.shape
    sums  = np.zeros(n_cols, np.float64)
    sumsq = np.zeros(n_cols, np.float64)
    counts  = np.zeros(n_cols, np.int64)

    for i in range(n_rows):
        for j in range(n_cols):
            v = X[i, j]
            if v != 0.0:
                sums[j]  += v
                sumsq[j] += v * v
                counts[j]  += 1

    mean = np.zeros(n_cols, np.float64)
    var  = np.zeros(n_cols, np.float64)
    for j in nb.prange(n_cols):
        c = counts[j]
        if c > 0:
            mu = sums[j] / c
            mean[j] = mu
            var[j]  = sumsq[j] / c - mu * mu
    return mean, var



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

# FIXME MICHELLE remove this for the DGE free version of code
def z_scale_genes_by_group(
    adata: ad.AnnData, 
    z_scale_group_col: Optional[str]
) -> np.ndarray:
    """
    Calculate z-scores for genes, optionally grouped by a column in `obs`,
    for use with a t-test. 
    """
    # FIXME MICHELLE: confirm that only non-zero values used for and scaled?

    if isinstance(adata.X, np.ndarray):
        converted_to_sparse = True
        X_zscaled = csr_matrix(adata.X) # FIXME MICHELLE: does this require a csr?
    else:
        converted_to_sparse = False
        X_zscaled = adata.X.copy() # FIXME MICHELLE: if we don't return data, can skip this copy

    if z_scale_group_col:
        # These are the row numbers of the cells to be z-scaled
        z_scale_indexes = {
            group: indices
            for group, indices in adata.obs.groupby(
                z_scale_group_col
            ).indices.items()
        }
    else:
        z_scale_indexes = {0: np.arange(len(adata))}

    logger.info(f"Z-scaling genes for {len(z_scale_indexes)} groups as defined by {z_scale_group_col}")
    n_rows, n_cols = X_zscaled.shape
    for group, indices in z_scale_indexes.items():
        gene_mean, gene_var = sparse_mean_var_minor_axis(
            X_zscaled.data,
            X_zscaled.indices,
            X_zscaled.indptr,
            n_rows,
            n_cols,
            CPU_COUNT,
        )
        gene_std = np.sqrt(gene_var)
        X_zscaled[indices] = (X_zscaled[indices] - gene_mean) / gene_std

    if converted_to_sparse:
        X_zscaled = X_zscaled.toarray()

    return X_zscaled

# FIXME MICHELLE remove this for the DGE free version of code
def preprocess_adata_for_deg(
    adata: ad.AnnData,
    deg_test_name: Literal["wilcoxon", "t-test"],
    filter_min_cells: int,
    filter_min_genes: int,
    z_scale_group_col: Optional[str],
) -> ad.AnnData:
    """
    Apply preprocessing to `adata` prior to DGE analysis based on the selected test.

    - For Wilcoxon: library size normalization and log1p transform.
    - For t-test: z-scale genes optionally by groups defined in `z_scale_group_col`.
    - Filter genes and cells by provided thresholds, with logging.
    """
    if adata.isbacked:
        adata = adata.to_memory()

    # Filter data
    # FIXME MICHELLE: should this be done before the scaling and transformation?
    n_vars = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=filter_min_cells)
    if n_vars != adata.n_vars:
        logger.info(
            f"Filtered {n_vars - adata.n_vars} genes using "
            f"min_cells={filter_min_cells}"
        )

    n_obs = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=filter_min_genes)
    if n_obs != adata.n_obs:
        logger.info(
            f"Filtered {n_obs - adata.n_obs} cells using "
            f"min_genes={filter_min_genes}"
        )

    # Transform data
    if deg_test_name == "wilcoxon":
        logger.info(
            "Normalizing total counts and log transforming for Wilcoxon test"
        )
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    else:
        # t-test
        if z_scale_group_col is None:
            log_message = (
                "Z-scaling genes by group for t-test using all cells"
            )
        else:
            log_message = (
                "Z-scaling genes by group for t-test using metadata "
                f"from column {z_scale_group_col}"
            )
        logger.info(log_message)
        X_zscaled = z_scale_genes_by_group(adata, z_scale_group_col)
        adata.X = X_zscaled

    # # FIXME MICHELLE: should this be done before the scaling and transformation?
    # n_vars = adata.n_vars
    # sc.pp.filter_genes(adata, min_cells=filter_min_cells)
    # if n_vars != adata.n_vars:
    #     logger.info(
    #         f"Filtered {n_vars - adata.n_vars} genes using "
    #         f"min_cells={filter_min_cells}"
    #     )

    # n_obs = adata.n_obs
    # sc.pp.filter_cells(adata, min_genes=filter_min_genes)
    # if n_obs != adata.n_obs:
    #     logger.info(
    #         f"Filtered {n_obs - adata.n_obs} cells using "
    #         f"min_genes={filter_min_genes}"
    #     )
    return adata


def run_multicondition_dge_analysis(
    adata: ad.AnnData,
    condition_key: str,
    control_name: str,
    control_cells_ids: Dict[str, List[str]],
    deg_test_name: Literal["wilcoxon", "t-test"] = "wilcoxon",
    filter_min_cells: int = 10, # FIXME MICHELLE: should these defaults be 0,0,1?
    filter_min_genes: int = 1000,
    min_pert_cells: int = 50,
    remove_avg_zeros: bool = False,
    return_merged_adata: bool = False,
    z_scale_group_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, ad.AnnData]:
    """
    z_scale_group_col: Optional[str] = None,
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
    z_scale_group_col (Optional[str], optional): Column name in `adata.obs` to use as groups for z-scaling. Only used for t-test. If None, all cells are used for z-scaling. Defaults to None.

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

    if deg_test_name == 'wilcoxon' and z_scale_group_col is not None:
        logger.warning(
            f"z_scale_group_col is set to {z_scale_group_col} but will not be used for Wilcoxon Rank-Sum test."
        )

    if deg_test_name == "t-test":
        if z_scale_group_col is None:
            logger.info(
                    "t-test is selected and z_scale_group_col is None, so all cells will be used for z-scaling."
                )
        else:
            logger.info(
                f"t-test is selected and z_scale_group_col is set to group cells by metadata from column {z_scale_group_col}."
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

    # FIXME MICHELLE: should this be done before the condition loop
    # adata = preprocess_adata_for_deg(
    #     adata=adata,
    #     deg_test_name=deg_test_name,
    #     filter_min_cells=filter_min_cells,
    #     filter_min_genes=filter_min_genes,
    #     z_scale_group_col=z_scale_group_col,
    # )

    target_conditions = list(control_cells_ids.keys())
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
            # FIXME MICHELLE: how to save unprocessed adata_merged?
            # if return_merged_adata:
            #     adata_merged.raw = adata_merged.copy()

            # Add simple comparison group label for rank_genes_groups
            comparison_group_col = "comparison_group"
            adata_merged.obs[comparison_group_col] = selected_condition
            control_idx = np.arange(num_condition, adata_merged.n_obs)
            adata_merged.obs.iloc[
                control_idx, adata_merged.obs.columns.get_loc(comparison_group_col)
            ] = control_name

            # Normalize and filter
            # FIXME MICHELLE: should this be done before the condition loop?
            if deg_test_name == "wilcoxon":
                logger.info(
                    "Normalizing total counts and log transforming for Wilcoxon test"
                )
                sc.pp.normalize_total(adata_merged, target_sum=1e4)
                sc.pp.log1p(adata_merged)

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
                # FIXME MICHELLE: how to save unprocessed adata_merged?
                # if adata_merged.raw is not None:
                #     adata_merged.X = adata_merged.raw.X
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

