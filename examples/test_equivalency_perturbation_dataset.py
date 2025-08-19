import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import anndata as ad
import shutil
import tempfile
import yaml
import os
from czbenchmarks.datasets.single_cell_perturbation import SingleCellPerturbationDataset
from czbenchmarks.constants import RANDOM_SEED
from czbenchmarks.datasets.utils import load_dataset
from tqdm import tqdm
from pandas.testing import assert_frame_equal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def create_adata(adata, target_gene_dict, nontargeting_cells):
    # Initialize list to store merged data
    all_merged_data = []

    # Initialize a dictionary to store target genes for each cell
    target_genes_to_save = {}

    target_genes = target_gene_dict.keys()
    for key in tqdm(target_genes, desc="Processing conditions"):
        adata_condition = adata[adata.obs["gene"] == key].copy()
        adata_control = adata[adata.obs.index.isin(nontargeting_cells[key])].copy()
        adata_condition.obs["condition"] = key
        adata_control.obs["condition"] = "non-targeting_" + key

        # Check if condition and control data have the same length
        if len(adata_condition) != len(adata_control):
            print(
                f"Warning: Condition and control data for {key} have different lengths."
            )
            continue

        # Merge condition and control data (avoid deprecated .concatenate and duplication warnings)
        adata_merged = ad.concat([adata_condition, adata_control], index_unique=None)

        # Add new column cell_barcode_gene
        adata_merged.obs["cell_barcode_gene"] = (
            adata_merged.obs.index.astype(str) + "_" + [key] * len(adata_merged)
        )

        # Add target genes to the dictionary for each cell
        for idx in adata_merged.obs["cell_barcode_gene"]:
            target_genes_to_save[idx] = target_gene_dict[key]

        all_merged_data.append(adata_merged)

    # Combine all adata objects using anndata.concat, auto-uniquifying obs names
    adata_final = ad.concat(all_merged_data, index_unique=None)

    # Set the new index
    adata_final.obs.set_index("cell_barcode_gene", inplace=True)
    return adata_final, target_genes_to_save


def sample_genes(
    df,
    percent_genes_to_mask,
    min_de_genes_to_mask,
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
        if n_genes_to_sample >= min_de_genes_to_mask:
            sampled_genes = np.random.choice(
                gene_names, size=n_genes_to_sample, replace=False
            ).tolist()
            target_gene_dict[target] = sampled_genes
    return target_gene_dict


def run_notebook_code(args):
    # These are the same numbers as in single_cell_perturbation.py

    adata = ad.read_h5ad(args.h5ad_data_path)
    df = pd.read_csv(args.de_results_path)
    if args.metric == "wilcoxon":
        df = df[np.abs(df["logfoldchanges"]) >= args.min_logfoldchange]
        df = df[df["pvals_adj"] < args.pval_threshold]
        target_gene_dict = sample_genes(
            df,
            args.percent_genes_to_mask,
            args.min_de_genes_to_mask,
            "target_gene",
            "ensembl_id",
        )

    elif args.metric == "t-test":
        df = df[np.abs(df["smd"]) >= args.min_smd]
        df = df[df["pval_adj"] < args.pval_threshold]
        target_gene_dict = sample_genes(
            df,
            args.percent_genes_to_mask,
            args.min_de_genes_to_mask,
            "condition",
            "gene",
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
        default="/data2/czbenchmarks/replogle2022/K562/K562_essential_raw_singlecell_01.h5ad",
        help="Path to masked h5ad file",
    )
    parser.add_argument(
        "--de_results_path",
        type=str,
        default="/data2/czbenchmarks/replogle2022/K562/zero_shot_benchmark/{metric_type}/de_results.csv",
        help="Path to de_results.csv file",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="wilcoxon",  # wilcoxon or t-test, do not use t_test
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
        "--min_de_genes_to_mask",
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
        default="/data2/czbenchmarks/replogle2022/K562/zero_shot_benchmark/ReplogleEssentialsCr4_GEM_libsizeMatched_NonTargetingCellIdsPerTarget.json",
        help="Path to control_cells_ids .json file",
    )
    parser.add_argument(
        "--run_notebook_code",
        action="store_true",
        help="Run notebook code",
    )
    args = parser.parse_args()
    # metric is either wilcoxon or t-test
    if args.metric not in {"wilcoxon", "t-test"}:
        raise ValueError(
            f"Unsupported --metric value: {args.metric}. Use 'wilcoxon' or 't-test'."
        )
    # metric_normalized is either wilcoxon or t_test
    metric_normalized = args.metric.strip().lower().replace("-", "_")
    if metric_normalized in {"t-test", "ttest"}:
        metric_normalized = "t_test"

    args.de_results_path = args.de_results_path.format(metric_type=metric_normalized)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nb_dir = Path(f"notebook_task_inputs_{metric_normalized}")

    #### NOTEBOOK CODE ####
    if args.run_notebook_code:
        if Path(nb_dir).exists():
            logger.info(
                f"Directory {nb_dir} already exists, copying to {nb_dir}_{timestamp}"
            )
            shutil.copytree(nb_dir, f"{nb_dir}_{timestamp}")
        else:
            logger.info(f"Directory {nb_dir} does not exist, creating it")
            os.makedirs(nb_dir)

    # This could be combined with previous if statement
    if (
        not args.run_notebook_code
        and Path(nb_dir / "notebook_adata_masked.h5ad").exists()
    ):
        logger.info(f"Loading notebook code from disk {nb_dir}")
        notebook_adata_masked = ad.read_h5ad(nb_dir / "notebook_adata_masked.h5ad")
        with open(nb_dir / "notebook_target_genes_to_save.json", "r") as f:
            notebook_target_genes_to_save = json.load(f)
        with open(nb_dir / "gene_map.json", "r") as f:
            gene_map = json.load(f)
    else:
        logger.info(f"Running notebook code with args: {args}")
        notebook_adata_masked, notebook_target_genes_to_save, gene_map = (
            run_notebook_code(args)
        )

        logger.info("Saving notebook code to disk")
        notebook_adata_masked.write(nb_dir / "notebook_adata_masked.h5ad")
        with (nb_dir / "notebook_target_genes_to_save.json").open("w") as f:
            json.dump(notebook_target_genes_to_save, f)
        with (nb_dir / "gene_map.json").open("w") as f:
            json.dump(gene_map, f)
        print(f"Notebook task inputs saved to: {nb_dir}")

    #### NEW CODE ####
    logger.info(f"Running new code with args: {args}")
    new_dataset = run_new_code(args.percent_genes_to_mask, args.metric)
    new_output_dir = new_dataset.store_task_inputs()
    logger.info(f"New code output saved to: {new_output_dir}")

    #### COMPARE OUTPUTS ####
    # Compare DE results CSV to dataset.de_results with column name mapping
    logger.info(
        "Comparing DE results CSV to dataset.de_results with column name mapping"
    )
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
    elif args.metric == "t-test":
        col_map = {
            "gene": "gene_id",
            "condition": "condition_name",
            "score": "score",
            "logfoldchange": "logfoldchange",
            "pval": "pval",
            "pval_adj": "pval_adj",
            "smd": "standardized_mean_diff",
        }

    df_csv = df_csv.rename(columns=col_map)
    filter = df_csv["pval_adj"] <= args.pval_threshold
    if metric_normalized == "wilcoxon":
        filter &= df_csv["logfoldchange"].abs() >= args.min_logfoldchange
    elif metric_normalized == "t_test":
        filter &= df_csv["standardized_mean_diff"].abs() >= args.min_smd

    df_csv = df_csv[filter]

    assert_de_results_equivalent(df_csv, new_dataset.de_results, col_map)
    logger.info("DE results matched")

    # Assert that the var frames are equivalent
    logger.info("Asserting that the var frames are equivalent")
    column_map = {
        "gene_name": "gene",
    }
    assert (notebook_adata_masked.var.index == new_dataset.adata.var.index).all()
    logger.info("Var frames matched")

    # Compare target genes
    logger.info("Comparing target genes")
    dataset_target_genes = {}
    for k in new_dataset.target_conditions_to_save.keys():
        s = k.split("_")
        try:
            dataset_target_genes[s[0] + "_" + gene_map[s[1]]] = (
                new_dataset.target_conditions_to_save[k]
            )
        except KeyError:
            print(f"KeyError: {k}")
            breakpoint()

    missing_keys = [
        k for k in dataset_target_genes.keys() if k not in notebook_target_genes_to_save
    ]
    assert not missing_keys, (
        f"Missing keys in notebook_target_genes_to_save: {missing_keys[:10]}"
    )
    for k, v in dataset_target_genes.items():
        v_lib = list(v)
        v_nb = list(notebook_target_genes_to_save[k])
        assert set(v_lib) == set(v_nb), f"Mismatched genes for key {k}"
    logger.info("Target genes matched")

    # Compare control matched adata
    logger.info("Comparing control matched adata")
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
        assert sorted(new_control_list) == sorted(
            list(nb_control)
        )  # order is suddenly not preserved
    logger.info("Control matched adata matched")

    logger.info("Done")
