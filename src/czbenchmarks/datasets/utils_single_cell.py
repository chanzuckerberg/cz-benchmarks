from typing import Tuple
import logging
import anndata as ad
import numpy as np
import numba as nb

logger = logging.getLogger(__name__)


@nb.njit(parallel=True, fastmath=True)
def colwise_nonzero_mean_var_numba(X):
    n_rows, n_cols = X.shape
    sums = np.zeros(n_cols, np.float64)
    sumsq = np.zeros(n_cols, np.float64)
    counts = np.zeros(n_cols, np.int64)

    for i in range(n_rows):
        for j in range(n_cols):
            v = X[i, j]
            if v != 0.0:
                sums[j] += v
                sumsq[j] += v * v
                counts[j] += 1

    mean = np.zeros(n_cols, np.float64)
    var = np.zeros(n_cols, np.float64)
    for j in nb.prange(n_cols):
        c = counts[j]
        if c > 0:
            mu = sums[j] / c
            mean[j] = mu
            var[j] = sumsq[j] / c - mu * mu
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
