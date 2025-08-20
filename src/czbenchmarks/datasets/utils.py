import os
import logging
import yaml
import hydra
from hydra.utils import instantiate
from typing import Dict, Optional, Tuple, List, Literal
import numpy as np
import scanpy as sc
import pandas as pd
import anndata as ad

from omegaconf import OmegaConf
from czbenchmarks.datasets.dataset import Dataset
from czbenchmarks.utils import initialize_hydra
from czbenchmarks.file_utils import download_file_from_remote

log = logging.getLogger(__name__)


def load_dataset(
    dataset_name: str,
    config_path: Optional[str] = None,
) -> Dataset:
    """
    Load, download (if needed), and instantiate a dataset using Hydra configuration.

    Args:
        dataset_name (str): Name of the dataset as specified in the configuration.
        config_path (Optional[str]): Optional path to a custom config YAML file. If not provided,
            only the package's default config is used.

    Returns:
        Dataset: Instantiated dataset object with data loaded.

    Raises:
        FileNotFoundError: If the custom config file does not exist.
        ValueError: If the specified dataset is not found in the configuration.

    Notes:
        - Merges custom config with default config if provided.
        - Downloads dataset file if a remote path is specified using `download_file_from_remote`.
        - Uses Hydra for instantiation and configuration management.
        - The returned dataset object is an instance of the `Dataset` class or its subclass.
    """
    initialize_hydra()

    # Load default config first and make it unstructured
    cfg = OmegaConf.create(
        OmegaConf.to_container(hydra.compose(config_name="datasets"), resolve=True)
    )

    # If custom config provided, load and merge it
    if config_path is not None:
        # Expand user path (handles ~)
        config_path = os.path.expanduser(config_path)
        config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Custom config file not found: {config_path}")

        # Load custom config
        with open(config_path) as f:
            custom_cfg = OmegaConf.create(yaml.safe_load(f))

        # Merge configs
        cfg = OmegaConf.merge(cfg, custom_cfg)

    if dataset_name not in cfg.datasets:
        raise ValueError(f"Dataset {dataset_name} not found in config")

    dataset_info = cfg.datasets[dataset_name]

    # Handle local caching and remote downloading
    dataset_info["path"] = download_file_from_remote(dataset_info["path"])

    # Instantiate the dataset using Hydra
    dataset = instantiate(dataset_info)

    # Load the dataset into memory
    dataset.load_data()

    return dataset


def list_available_datasets() -> Dict[str, Dict[str, str]]:
    """
    Return a sorted list of all dataset names defined in the `datasets.yaml` Hydra configuration.

    Returns:
        List[str]: Alphabetically sorted list of available dataset names.

    Notes:
        - Loads configuration using Hydra.
        - Extracts dataset names from the `datasets` section of the configuration.
        - Sorts the dataset names alphabetically for easier readability.
    """
    initialize_hydra()

    # Load the datasets configuration
    cfg = OmegaConf.to_container(hydra.compose(config_name="datasets"), resolve=True)

    # Extract dataset names
    datasets = {
        name: {
            "organism": str(dataset_info.get("organism", "Unknown")),
            "url": dataset_info.get("path", "Unknown"),
        }
        for name, dataset_info in cfg.get("datasets", {}).items()
    }

    # Sort alphabetically for easier reading
    datasets = dict(sorted(datasets.items()))

    return datasets


def run_multicondition_dge_analysis(
    adata: ad.AnnData,
    condition_key: str,
    control_cells_ids: Dict[str, List[str]],
    deg_test_name: Literal["wilcoxon", "t-test"] = "wilcoxon",
    filter_min_cells: int = 10,
    filter_min_genes: int = 1000,  # FIXME MICHELLE: ask Jasleen about default
    min_pert_cells: int = 50,  # FIXME MICHELLE: ask Jasleen about default
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
    control_cells_ids (Dict[str, List[str]]): Mapping from condition -> list of matched control cell ids.
    deg_test_name (Literal["wilcoxon", "t-test"], optional): Statistical test name for differential expression. Defaults to 'wilcoxon'.
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

    if deg_test_name not in ["wilcoxon", "t-test"]:
        raise ValueError(
            f"Invalid deg_test_name: {deg_test_name}. Must be 'wilcoxon' or 't-test'."
        )

    if return_merged_adata:
        log.warning("return_merged_adata is True, which can consume a lot of memory.")

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
            print(f"Insufficient cells for analysis of {selected_condition}")
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
                f"remove_avg_zeros is True.Removing {len(results) - len(indexes)} genes with zero expression"
            )
            results = results.iloc[indexes]

        results_df.append(results)
        if return_merged_adata:
            adata_results.append(adata_merged)

    results = pd.concat(results_df, ignore_index=True)
    del results_df

    dge_params = adata_merged.uns["dge_results"]["params"]
    if return_merged_adata:
        adata_merged = ad.concat(adata_results, index_unique=None)
        del adata_results
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
    results = results.rename(columns=col_mapper)
    cols = [x for x in col_mapper.values() if x in results.columns]
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


