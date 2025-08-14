import argparse
import json
import numpy as np
import pandas as pd
import scanpy as sc
import tempfile
import yaml
from czbenchmarks.datasets.single_cell_perturbation import SingleCellPerturbationDataset
from czbenchmarks.constants import RANDOM_SEED
from czbenchmarks.datasets.utils import load_dataset
from tqdm import tqdm
from pandas.testing import assert_frame_equal


def assert_de_results_equivalent(df1, df2, col_map):
    """
    Compare two DE results DataFrames with column name mapping.
    Asserts that the mapped DataFrames are nearly equal.
    """
    df1_mapped = df1.rename(columns=col_map)
    common_cols = sorted(set(df1_mapped.columns) & set(df2.columns))
    left = df1_mapped[common_cols].sort_values(by=common_cols).reset_index(drop=True)
    right = df2[common_cols].sort_values(by=common_cols).reset_index(drop=True)
    assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)


def assert_var_equivalent(var1, var2, col_map):
    """
    Compare two AnnData.var DataFrames with column name mapping.
    Asserts that the mapped DataFrames are nearly equal on common columns.
    """
    var1_mapped = var1.rename(columns=col_map)
    common_cols = sorted(set(var1_mapped.columns) & set(var2.columns))
    if not common_cols:
        raise AssertionError(
            "No common columns in var DataFrames after applying column map"
        )
    left = var1_mapped[common_cols].sort_values(by=common_cols).reset_index(drop=True)
    right = var2[common_cols].sort_values(by=common_cols).reset_index(drop=True)
    assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)


def create_adata(adata, target_gene_dict, nontargeting_cells):
    # Initialize list to store merged data
    all_merged_data = []

    # Initialize a dictionary to store target genes for each cell
    target_genes_to_save = {}

    target_genes = target_gene_dict.keys()
    for key in tqdm(target_genes, desc="Processing conditions"):
        adata_condition = adata[adata.obs["gene"] == key]
        adata_control = adata[adata.obs.index.isin(nontargeting_cells[key])]
        adata_condition.obs["condition"] = key
        adata_control.obs["condition"] = "non-targeting_" + key

        # Check if condition and control data have the same length
        if len(adata_condition) != len(adata_control):
            print(
                f"Warning: Condition and control data for {key} have different lengths."
            )
            continue

        # Merge condition and control data
        adata_merged = adata_condition.concatenate(adata_control, index_unique=None)

        # Add new column cell_barcode_gene
        adata_merged.obs["cell_barcode_gene"] = (
            adata_merged.obs.index.astype(str) + "_" + [key] * len(adata_merged)
        )

        # Add target genes to the dictionary for each cell
        for idx in adata_merged.obs["cell_barcode_gene"]:
            target_genes_to_save[idx] = target_gene_dict[key]

        all_merged_data.append(adata_merged)

    # Combine all adata objects
    adata_final = all_merged_data[0].concatenate(all_merged_data[1:], index_unique=None)

    # Set the new index
    adata_final.obs.set_index("cell_barcode_gene", inplace=True)
    return adata_final, target_genes_to_save


def sample_genes(
    df,
    percent_genes_to_mask,
    min_de_genes,
    condition_col,
    gene_col,
    seed: int = RANDOM_SEED,
):
    np.random.seed(seed)
    target_genes = df[condition_col].unique()
    target_gene_dict = {}
    for target in tqdm(target_genes):
        gene_names = df[df[condition_col] == target][gene_col].values
        n_genes_to_sample = int(len(gene_names) * percent_genes_to_mask)
        if n_genes_to_sample >= min_de_genes:
            sampled_genes = np.random.choice(
                gene_names, size=n_genes_to_sample, replace=False
            ).tolist()
            target_gene_dict[target] = sampled_genes
    return target_gene_dict


def run_notebook_code(args):
    # These are the same numbers as in single_cell_perturbation.py

    adata = sc.read_h5ad(args.h5ad_data_path)
    df = pd.read_csv(args.de_results_path)
    if args.metric == "wilcoxon":
        df = df[np.abs(df["logfoldchanges"]) >= args.min_logfoldchange]
        df = df[df["pvals_adj"] < args.pval_threshold]
        target_gene_dict = sample_genes(
            df,
            args.percent_genes_to_mask,
            args.min_de_genes,
            "target_gene",
            "ensembl_id",
        )

    elif args.metric == "t_test":
        df = df[np.abs(df["smd"]) >= args.min_smd]
        df = df[df["pval_adj"] < args.pval_threshold]
        target_gene_dict = sample_genes(
            df, args.percent_genes_to_mask, args.min_de_genes, "condition", "gene"
        )

    else:
        raise ValueError(f"Metric {args.metric} not supported")

    with open(args.control_cells_ids_path, "r") as f:
        nontargeting_cells = json.load(f)
    adata_final, target_genes_to_save = create_adata(
        adata, target_gene_dict, nontargeting_cells
    )
    gene_map = {}
    for index, ent in adata.obs.iterrows():
        gene_map[ent.gene_id] = ent.gene

    return adata_final, target_genes_to_save, gene_map


def run_new_code(
    percent_genes_to_mask: float, metric: str
) -> SingleCellPerturbationDataset:
    # Create a temporary Hydra config to pass through percent_genes_to_mask
    custom_config = {
        "datasets": {
            "replogle_k562_essential_perturbpredict": {
                "percent_genes_to_mask": percent_genes_to_mask,
                "deg_test_name": metric,
            }
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp_cfg:
        yaml.dump(custom_config, tmp_cfg)
        tmp_cfg_path = tmp_cfg.name

    dataset: SingleCellPerturbationDataset = load_dataset(
        "replogle_k562_essential_perturbpredict", config_path=tmp_cfg_path
    )
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--percent_genes_to_mask",
        type=float,
        default=1.0,
        help="Percentage of genes to mask",
    )
    parser.add_argument(
        "--h5ad_data_path",
        type=str,
        default="new_data/K562_essential_raw_singlecell_01.h5ad",
        help="Path to masked h5ad file",
    )
    parser.add_argument(
        "--de_results_path",
        type=str,
        default="k562_data/wilcoxon_de_results.csv",
        help="Path to de_results.csv file",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="wilcoxon",
        help="Metric to use for DE analysis",
    )
    parser.add_argument(
        "--min_logfoldchange",
        type=float,
        default=1.0,
        help="Minimum absolute log-fold change for DE filtering (used when --metric=wilcoxon)",
    )
    parser.add_argument(
        "--pval_threshold",
        type=float,
        default=1e-4,
        help="Adjusted p-value threshold for DE filtering",
    )
    parser.add_argument(
        "--min_de_genes",
        type=int,
        default=5,
        help="Minimum number of DE genes required to mask a condition",
    )
    parser.add_argument(
        "--min_smd",
        type=float,
        default=0.55,
        help="Minimum standardized mean difference for DE filtering (used when --metric=t-test)",
    )
    parser.add_argument(
        "--control_cells_ids_path",
        type=str,
        default="new_data/ReplogleEssentialsCr4_GEM_libsizeMatched_NonTargetingCellIdsPerTarget.json",
        help="Path to control_cells_ids .json file",
    )
    args = parser.parse_args()
    # Normalize metric aliases (accepts t_test or t-test, but preserve t_test for dataset)
    metric_normalized = args.metric.strip().lower().replace("-", "_")
    if metric_normalized in {"t_test", "ttest"}:
        args.metric = "t_test"
    elif metric_normalized == "wilcoxon":
        args.metric = "wilcoxon"
    else:
        raise ValueError(
            f"Unsupported --metric value: {args.metric}. Use 'wilcoxon' or 't_test'."
        )
    notebook_adata_masked, notebook_target_genes_to_save, gene_map = run_notebook_code(
        args
    )
    new_dataset = run_new_code(args.percent_genes_to_mask, args.metric)

    # Compare DE results CSV to dataset.de_results with column name mapping
    df_csv = pd.read_csv(args.de_results_path)
    if args.metric == "wilcoxon":
        col_map = {
            "names": "gene_id",
            "target_gene": "condition_name",
            "scores": "score",
            "logfoldchanges": "logfoldchange",
            "pvals": "pval",
            "pvals_adj": "pval_adj",
        }
    elif args.metric == "t_test":
        col_map = {
            "gene": "gene_id",
            "condition": "condition_name",
            "score": "score",
            "logfoldchange": "logfoldchange",
            "pval": "pval",
            "pval_adj": "pval_adj",
            "smd": "standardized_mean_diff",
        }
    assert_de_results_equivalent(df_csv, new_dataset.de_results, col_map)

    # Assert that the var frames are equivalent
    column_map = {
        "gene_name": "gene",
    }
    assert_var_equivalent(notebook_adata_masked.var, new_dataset.adata.var, column_map)

    dataset_target_genes = {}
    for k in new_dataset.target_genes_to_save.keys():
        s = k.split("_")
        try:
            dataset_target_genes[s[0] + "_" + gene_map[s[1]]] = (
                new_dataset.target_genes_to_save[k]
            )
        except KeyError:
            print(f"KeyError: {k}")
            breakpoint()

    missing_keys = [
        k for k in dataset_target_genes.keys() if k not in notebook_target_genes_to_save
    ]
    assert (
        not missing_keys
    ), f"Missing keys in notebook_target_genes_to_save: {missing_keys[:10]}"
    for k, v in dataset_target_genes.items():
        v_lib = list(v)
        v_nb = list(notebook_target_genes_to_save[k])
        assert set(v_lib) == set(v_nb), f"Mismatched genes for key {k}"

    control_prefix = "non-targeting"
    # we want to make sure that the conditions and cell_bar_code_gene are the same
    new_obs = new_dataset.control_matched_adata.obs
    notebook_obs = notebook_adata_masked.obs
    condition_series = new_obs["condition"].astype(str)
    condition_list = np.unique(
        condition_series[~condition_series.str.startswith(control_prefix)]
    )
    for condition in condition_list:
        new_condition = new_obs[new_obs["condition"] == condition].index
        new_control = new_obs[
            new_obs["condition"] == f"{control_prefix}_{condition}"
        ].index
        condition_name = gene_map[condition]

        all_condition = notebook_obs[
            notebook_obs.index.str.endswith(f"_{condition_name}")
        ]
        nb_condition = all_condition[all_condition["gene"] != "non-targeting"].index
        nb_control = all_condition[all_condition["gene"] == "non-targeting"].index
        new_control_list = []
        for k in new_control:
            s = k.split("_")
            new_control_list.append(s[0] + "_" + gene_map[s[1]])
        new_condition_list = []
        for k in new_condition:
            s = k.split("_")
            new_condition_list.append(s[0] + "_" + gene_map[s[1]])

        assert new_condition_list == list(nb_condition)
        assert new_control_list == list(nb_control)
