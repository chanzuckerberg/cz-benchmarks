from typing import Tuple
import logging
import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)


def create_adata_for_condition(
    adata: ad.AnnData,
    condition: str,
    condition_key: str,
    control_name: str,
    condition_control_sep: str,
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
        condition_control_sep: str, separator between control name and condition
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
    label_ctrl = [f"{control_name}{condition_control_sep}{condition}"] * len(
        adata_control
    )
    adata_merged.obs[condition_key] = label_cond + label_ctrl

    # Add condition to cell_barcode_gene column and set as index
    adata_merged.obs_names = (
        adata_merged.obs_names.astype(str) + condition_control_sep + condition
    )

    return adata_merged
