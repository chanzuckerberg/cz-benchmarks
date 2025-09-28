"""Generic single-cell preprocessing + control cell matching + DGE pipeline.

Flow:
raw .h5ad
  -> dataset-specific preprocess (e.g., replogle_preprocessor)
  -> compute UMI_count/ngenes
  -> matched controls per condition
  -> run_multicondition_dge_analysis (Wilcoxon)
  -> write processed .h5ad with added UNS keys

This script wires together:
- Dataset-specific preprocessing (imported from a sibling module like
  `replogle_preprocessor.py`)

    # NOTE: Dataset-specific preprocessing is implemented in a sibling module
    # (e.g., replogle_preprocessor.py). In the future, we can generalize this to a
    # plugin folder and dynamic imports, but for now we keep it local and explicit.

- Matched control cell selection per perturbation condition
- Multi-condition differential expression (Wilcoxon) via czbenchmarks utils

Inputs: a raw .h5ad file. Outputs: a processed .h5ad with the following
AnnData.uns fields stored:

- control match map: `adata.uns[UNS_KEY_CONTROL_CELLS_MAP]`
- control id list per condition: `adata.uns[UNS_KEY_CONTROL_CELLS_IDS]`
- differential expression (Wilcoxon): `adata.uns[UNS_KEY_DE_RESULTS_WILCOXON]`

Note: These constants may be unified with the corresponding definitions in the
`SingleCellPerturbation` dataset fields to avoid duplication and ensure
consistency across the codebase.
"""

import argparse
import os
from pathlib import Path
from typing import List, Dict, Callable, Optional

import numpy as np
import pandas as pd
import anndata as ad
from tqdm.auto import tqdm

from czbenchmarks.datasets.utils_single_cell import run_multicondition_dge_analysis
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import pairwise_distances


# AnnData.uns key constants
UNS_KEY_CONTROL_CELLS_MAP = "control_cells_map"
UNS_KEY_CONTROL_CELLS_IDS = "control_cells_ids"
UNS_KEY_DE_RESULTS_WILCOXON = "de_results_wilcoxon"

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _to_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    return pd.to_numeric(series, errors="coerce")


def _compute_counts_and_genes(adata: ad.AnnData) -> None:
    if hasattr(adata.X, "sum"):
        umi_count = np.asarray(adata.X.sum(axis=1)).ravel()
    else:
        umi_count = np.sum(adata.X, axis=1)
    adata.obs["UMI_count"] = umi_count

    if hasattr(adata.X, "todense") or hasattr(adata.X, "tocsc"):
        ngenes = np.asarray((adata.X > 0).sum(axis=1)).ravel()
    else:
        ngenes = np.count_nonzero(adata.X, axis=1)
    adata.obs["ngenes"] = ngenes

    adata.obs["UMI_count"] = pd.to_numeric(adata.obs["UMI_count"], errors="coerce")


def _compute_and_store_control_matches(
    adata: ad.AnnData,
    *,
    condition_col: str = "condition",
    control_name: str = "non-targeting",
    min_pert_cells: int = 50,
) -> Dict[str, List[str]]:
    conditions = (
        adata.obs[condition_col].astype(str).unique().tolist()
        if condition_col in adata.obs
        else []
    )
    conditions = [c for c in conditions if c != str(control_name)]

    control_cells_map: Dict[str, Dict[str, str]] = {}
    control_cells_ids: Dict[str, List[str]] = {}

    for cond in tqdm(conditions, desc="Matching controls", unit="cond"):
        matched = get_matched_controls(
            adata=adata,
            perturbation=cond,
            min_cells=min_pert_cells,
            verbose=False,
            dict_ctrls=None,
            pert_column=condition_col,
            ctrl_condition=control_name,
            gem_column="gem_group" if "gem_group" in adata.obs.columns else None,
            libsize_column="UMI_count" if "UMI_count" in adata.obs.columns else None,
            ngenes_column="ngenes" if "ngenes" in adata.obs.columns else None,
        )
        if isinstance(matched, dict) and len(matched) > 0:
            control_cells_map[cond] = matched
            control_cells_ids[cond] = list(matched.values())

    adata.uns[UNS_KEY_CONTROL_CELLS_MAP] = control_cells_map
    adata.uns[UNS_KEY_CONTROL_CELLS_IDS] = control_cells_ids

    return control_cells_ids


def _compute_de_wilcoxon(
    adata: ad.AnnData,
    condition_col: str = "condition",
    control_name: str = "non-targeting",
    *,
    filter_min_cells: int = 0,
    filter_min_genes: int = 0,
    min_pert_cells: int = 50,
) -> pd.DataFrame:
    if condition_col not in adata.obs.columns:
        raise ValueError(f"Missing '{condition_col}' in adata.obs for DE computation")
    if not adata.obs[condition_col].astype(str).str.contains(control_name).any():
        raise ValueError(
            f"Control '{control_name}' not found in adata.obs['{condition_col}']"
        )

    control_cells_ids = adata.uns.get(UNS_KEY_CONTROL_CELLS_IDS) if hasattr(adata, "uns") else None
    if not control_cells_ids:
        control_cells_ids = _compute_and_store_control_matches(
            adata,
            condition_col=condition_col,
            control_name=control_name,
            min_pert_cells=min_pert_cells,
        )

    if len(control_cells_ids) == 0:
        return pd.DataFrame(columns=["condition", "gene", "logfoldchange", "pval_adj"]).copy()

    results_df, _ = run_multicondition_dge_analysis(
        adata=adata,
        condition_key=condition_col,
        control_cells_ids=control_cells_ids,
        deg_test_name="wilcoxon",
        filter_min_cells=filter_min_cells,
        filter_min_genes=filter_min_genes,
        min_pert_cells=min_pert_cells,
        remove_avg_zeros=False,
        store_dge_metadata=False,
        return_merged_adata=False,
    )

    if results_df is None or len(results_df) == 0:
        return pd.DataFrame(columns=["condition", "gene", "logfoldchange", "pval_adj"]).copy()

    results_df = results_df.rename(columns={"gene_id": "gene"})
    keep_cols = ["condition", "gene", "logfoldchange", "pval_adj"]
    available = [c for c in keep_cols if c in results_df.columns]
    return results_df[available].copy()


def get_matched_controls(
    adata,
    pert_column: str,
    ctrl_condition: str,
    gem_column: str,
    libsize_column: str,
    *,
    perturbation=None,
    min_cells: int = 50,
    ngenes_column: str = None,
    feature_columns: List[str] = None,
    verbose: bool = False,
    dict_ctrls=None,
    obs_metadata: pd.DataFrame = None,
):
    """
    Get matched control cells for a given perturbation in an experiment.

    Returns a mapping from treated_cell_id -> matched_control_cell_id.
    """
    if any(field is None for field in [pert_column, ctrl_condition, gem_column, libsize_column]):
        raise ValueError("All column fields (pert_column, ctrl_condition, gem_column, libsize_column) must be provided")

    if obs_metadata is not None:
        obs_df = obs_metadata.copy()
    else:
        if adata is None:
            raise ValueError("Either adata or obs_metadata must be provided")
        obs_df = adata.obs.copy()

    if (dict_ctrls is not None) and (verbose) and (perturbation is not None):
        print(f"Using pre-computed matched controls for {perturbation}")
        return dict_ctrls[perturbation]

    if feature_columns is not None and len(feature_columns) > 0:
        feature_cols = feature_columns
    else:
        feature_cols = []
        if libsize_column is not None and libsize_column in obs_df.columns:
            feature_cols.append(libsize_column)
        if ngenes_column is not None and ngenes_column in obs_df.columns:
            feature_cols.append(ngenes_column)

    if any(col not in obs_df.columns for col in feature_cols):
        missing = [col for col in feature_cols if col not in obs_df.columns]
        raise ValueError(f"Missing feature columns in metadata: {missing}")

    ctrl_df = obs_df[obs_df[pert_column] == ctrl_condition].copy()

    if perturbation is None:
        raise ValueError("perturbation must be provided")
    condition = perturbation

    pert_df = obs_df[obs_df[pert_column] == condition].copy()
    if len(pert_df) < min_cells:
        print(f"Warning: Only {len(pert_df)} cells for {condition}")
        return None

    if gem_column is not None and gem_column in pert_df.columns:
        groups = list(pert_df.groupby(gem_column))
    else:
        groups = [(None, pert_df)]

    used_controls = set()
    matched_controls_map: Dict[str, str] = {}

    for group_name, group in groups:
        if len(group) == 0:
            continue

        if gem_column is not None and gem_column in ctrl_df.columns and group_name is not None:
            available_ctrls = ctrl_df[ctrl_df[gem_column] == group_name].copy()
        else:
            available_ctrls = ctrl_df.copy()

        available_ctrls = available_ctrls[~available_ctrls.index.isin(used_controls)]
        if len(available_ctrls) == 0:
            print(f"Warning: No unused controls available in group {group_name if group_name is not None else 'ALL'} for condition {condition}")
            continue

        treat_features = group[feature_cols].values.astype(float)
        ctrl_features = available_ctrls[feature_cols].values.astype(float)
        treat_features = np.log1p(np.maximum(treat_features, 0.0))
        ctrl_features = np.log1p(np.maximum(ctrl_features, 0.0))
        pooled = np.vstack([treat_features, ctrl_features])
        mu = pooled.mean(axis=0)
        sigma = pooled.std(axis=0)
        sigma[sigma < 1e-8] = 1.0
        treat_features = (treat_features - mu) / sigma
        ctrl_features = (ctrl_features - mu) / sigma
        dist_matrix = pairwise_distances(treat_features, ctrl_features)

        t_indices, c_indices = linear_sum_assignment(dist_matrix)
        max_pairs = min(len(t_indices), ctrl_features.shape[0])
        for i in range(max_pairs):
            treat_idx = t_indices[i]
            ctrl_idx = c_indices[i]
            matched_control = available_ctrls.index[ctrl_idx]
            matched_controls_map[group.index[treat_idx]] = matched_control
            used_controls.add(matched_control)

    return matched_controls_map


def run_pipeline(
    in_path: str,
    *,
    dataset_preprocess: Optional[Callable[[ad.AnnData], ad.AnnData]] = None,
    condition_col: str = "condition",
    control_name: str = "non-targeting",
    filter_min_cells: int = 0,
    filter_min_genes: int = 0,
    min_pert_cells: int = 50,
) -> ad.AnnData:
    adata = ad.read_h5ad(in_path, backed=None)

    if dataset_preprocess is not None:
        adata = dataset_preprocess(adata)

    if "UMI_count" not in adata.obs or "ngenes" not in adata.obs:
        _compute_counts_and_genes(adata)

    _compute_and_store_control_matches(
        adata,
        condition_col=condition_col,
        control_name=control_name,
        min_pert_cells=min_pert_cells,
    )

    de_df = _compute_de_wilcoxon(
        adata,
        condition_col=condition_col,
        control_name=control_name,
        filter_min_cells=filter_min_cells,
        filter_min_genes=filter_min_genes,
        min_pert_cells=min_pert_cells,
    )
    adata.uns[UNS_KEY_DE_RESULTS_WILCOXON] = de_df

    return adata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Generic single-cell preprocessing + DE pipeline. "
            "Dataset-specific preprocessing is imported from a sibling module."
        )
    )
    p.add_argument("--input_path", required=True, help="Path to a raw .h5ad file to process")
    p.add_argument("--output_dir", required=True, help="Directory to write the processed .h5ad file")
    p.add_argument("--condition_col", default="condition", help="Obs column name for perturbation condition")
    p.add_argument("--control_name", default="non-targeting", help="Control group name in condition column")
    p.add_argument("--filter_min_cells", type=int, default=10, help="Minimum cells per gene for filtering before DGE")
    p.add_argument("--filter_min_genes", type=int, default=1000, help="Minimum genes per cell for filtering before DGE")
    p.add_argument("--min_pert_cells", type=int, default=10, help="Minimum perturbed cells required per condition")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_dir(args.output_dir)

    in_path = args.input_path
    basename = os.path.basename(in_path)
    name, _ = os.path.splitext(basename)
    out_path = os.path.join(args.output_dir, f"{name}_processed.h5ad")

    # Import dataset-specific preprocessor locally and pass to the pipeline
    # this can be replaced with a dynamic import if we want to generalize this to a plugin folder
    # see note in the file header
    from replogle_preprocessor import preprocess as _dataset_preprocess

    adata = run_pipeline(
        in_path,
        dataset_preprocess=_dataset_preprocess,
        condition_col=args.condition_col,
        control_name=args.control_name,
        filter_min_cells=args.filter_min_cells,
        filter_min_genes=args.filter_min_genes,
        min_pert_cells=args.min_pert_cells,
    )
    adata.write_h5ad(out_path)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()


