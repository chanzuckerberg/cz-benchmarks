# This tests that the perturbation prediection task with the new data formats matches
# what is outputted in Maria's intial notebook.

import argparse
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
import json


def dict_numpy_to_list(d):
    # Convert numpy arrays to lists for JSON serialization
    def convert_value(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            return float(v)
        elif isinstance(v, (np.int32, np.int64)):
            return int(v)
        elif isinstance(v, list):
            return [convert_value(item) for item in v]
        else:
            return v

    return {k: convert_value(v) for k, v in d.items()}


def compute_log_fold_change(probs1, probs2, epsilon=1e-10, probs=True):
    # Add epsilon to avoid division by zero or log of zero
    if probs:
        probs1 = np.clip(probs1, epsilon, 1.0)
        probs2 = np.clip(probs2, epsilon, 1.0)

    # Compute log fold change
    log_fc = np.log2(probs1 / probs2)
    return log_fc


def run_notebook_code(args):
    # Read in and filter the DE results
    de_results = pd.read_csv(args.de_results_path)
    if args.metric_type == "wilcoxon":
        de_results = de_results[
            np.abs(de_results["logfoldchanges"]) >= args.min_logfoldchanges
        ]
        de_results = de_results[de_results["pvals_adj"] < args.pval_threshold]
    elif args.metric_type == "t_test":
        de_results = de_results[de_results["pval_adj"] < args.pval_threshold]
        de_results = de_results[
            de_results["smd"].abs() >= args.min_standard_mean_deviation
        ]
        de_results["target_gene"] = de_results["condition"]
        de_results["ensembl_id"] = de_results["gene"]
        de_results["names"] = de_results["gene"]

    # run the code that is basically in the notebooks!
    # load the data
    masked_adata = ad.read_h5ad(args.masked_h5ad_path)

    # The original code is per model name. But here, we just have one model.
    predictions_dict = {}
    genes_dict = {}
    pred_log_fc_dict = {}
    true_log_fc_dict = {}

    pred = np.load(args.predictions_path)
    sample_id = np.load(args.sample_id_path)
    target_genes = np.load(args.target_genes_path)
    df = pd.DataFrame(
        {"sample_id": sample_id, "pred": list(pred), "target_genes": list(target_genes)}
    )

    # Group predictions by sample
    for sample, group in tqdm(df.groupby("sample_id")):
        predictions_dict.setdefault(sample, []).append(
            np.concatenate(group["pred"].to_numpy())
        )
        if sample not in genes_dict:
            genes_dict[sample] = group["target_genes"].to_numpy()

    # Iterate over all conditions
    for condition in masked_adata.obs["gene"].unique():
        # Select appropriate DE results. For model predicting log1p counts/binary class probabilities we use wilcoxon test results and logFC.
        # For model predicting z-scale expression we use t-test results and SMD.
        condition_de_df = de_results[de_results["target_gene"] == condition]
        if len(condition_de_df) < args.min_de_genes:
            print(f"Not enough DE genes for condition {condition}. Skipping...")
            continue

        adata_condition = masked_adata[
            masked_adata.obs.index.str.endswith(f"_{condition}")
        ]
        condition_cells = adata_condition[
            adata_condition.obs["gene"] != "non-targeting"
        ].obs.index
        control_cells = adata_condition[
            adata_condition.obs["gene"] == "non-targeting"
        ].obs.index
        if len(condition_cells) == 0 or len(control_cells) == 0:
            print(
                f"No condition or control cells for condition {condition}. Skipping..."
            )
            continue

        if len(control_cells) != len(condition_cells):
            raise AssertionError(
                f"Number of control cells ({len(control_cells)}) is not equal to number of condition cells ({len(condition_cells)})"
            )
        if len(control_cells) < 10:
            print(f"Less than 10 cells in condition {condition}. Skipping...")
            continue

        try:
            # Masked genes for this sample
            masked_genes = genes_dict[control_cells[0]]
            mask = masked_genes != "A"
            masked_genes = masked_genes[mask]
            n_masked_genes = len(masked_genes)

            # Get predictions for control and condition cells
            control_predictions = np.asarray(
                [predictions_dict[cell][-1][:n_masked_genes] for cell in control_cells]
            )
            condition_predictions = np.asarray(
                [
                    predictions_dict[cell][-1][:n_masked_genes]
                    for cell in condition_cells
                ]
            )

            # For classification models, apply sigmoid
            # model_configs and model are not defined in this snippet, so this block may need to be adapted
            # if model_configs[model].model_type == "classification":
            #     control_predictions = sigmoid(control_predictions)
            #     condition_predictions = sigmoid(condition_predictions)

            # Initialize per-condition containers
            pred_log_fc_dict.setdefault(condition, [])
            true_log_fc_dict.setdefault(condition, [])
            # num_de_genes_dict is not used elsewhere, so we can comment or remove it
            # num_de_genes_dict.setdefault(condition, [])

            # Compute predicted log fold change
            if args.metric_type == "t_test":
                probs = False  # The model is a regression model
                pred_log_fc = compute_log_fold_change(
                    condition_predictions.mean(axis=0),
                    control_predictions.mean(axis=0),
                    probs=probs,
                )
            else:
                pred_log_fc = condition_predictions.mean(
                    axis=0
                ) - control_predictions.mean(axis=0)

            # Select column for true log fold change
            col = "smd" if args.metric_type == "t_test" else "logfoldchanges"
            # Align true log fold change to masked genes
            true_log_fc = (
                condition_de_df.set_index("names").reindex(masked_genes)[col].values
            )

            # Remove NaNs
            valid = ~np.isnan(true_log_fc)
            pred_log_fc = pred_log_fc[valid]
            true_log_fc = true_log_fc[valid]

            pred_log_fc_dict[condition].append(pred_log_fc)
            true_log_fc_dict[condition].append(true_log_fc)
        except Exception as e:
            print(f"Error processing condition {condition}: {e}")
            continue
    return pred_log_fc_dict, true_log_fc_dict


def run_new_code(args):
    task = PerturbationExpressionPredictionTask(
        min_de_genes=args.min_de_genes,
        min_logfoldchange=args.min_logfoldchanges,
        metric_type=args.metric_type,
        standardized_mean_diff=args.min_standard_mean_deviation,
    )

    adata = ad.read_h5ad(args.new_h5ad_path, backed="r")
    task_input = PerturbationExpressionPredictionTaskInput(
        de_results=adata.uns[f"de_results_{args.metric_type}"],
        control_cells_ids=adata.uns["control_cells_ids"],
    )

    # Load prediction data
    sample_id = np.load(args.sample_id_path)
    pred = np.load(args.predictions_path)
    target_genes = np.load(args.target_genes_path)

    pred_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "pred": list(pred),
            "target_genes": list(target_genes),
        }
    )

    result = task._run_task(pred_df, task_input)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run K562 Perturbation Task with custom file paths and filtering parameters."
    )
    parser.add_argument(
        "--min_logfoldchanges",
        type=float,
        default=1.0,
        help="Minimum absolute log fold change for DE gene filtering",
    )
    parser.add_argument(
        "--pval_threshold",
        type=float,
        default=1e-4,
        help="Adjusted p-value threshold for DE gene filtering",
    )
    parser.add_argument(
        "--new_h5ad_path",
        type=str,
        default="new_data/replogle_k562_essential_perturbpredict.h5ad",
        help="Path to masked h5ad file",
    )
    parser.add_argument(
        "--masked_h5ad_path",
        type=str,
        default="k562_data/zero_shot_0.5_de_genes_masked.h5ad",
        help="Path to masked h5ad file",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="k562_data/predictions_merged.npy",
        help="Path to predictions .npy file",
    )
    parser.add_argument(
        "--sample_id_path",
        type=str,
        default="k562_data/sample_id_merged.npy",
        help="Path to sample_id .npy file",
    )
    parser.add_argument(
        "--target_genes_path",
        type=str,
        default="k562_data/target_genes_merged.npy",
        help="Path to target_genes .npy file",
    )
    parser.add_argument(
        "--min_de_genes",
        type=int,
        default=20,
        help="Minimum number of DE genes for perturbation condition",
    )
    parser.add_argument(
        "--de_results_path",
        type=str,
        default="k562_data/wilcoxon_de_results.csv",
        help="Path to de_results .csv file",
    )
    parser.add_argument(
        "--t_test_de_results_path",
        type=str,
        default="k562_data/t_test_de_results.csv",
        help="Path to t_test de_results .csv file",
    )
    parser.add_argument(
        "--metric_type",
        type=str,
        choices=["wilcoxon", "t_test"],
        default="wilcoxon",
        help="Metric type to use for DE gene filtering: 'wilcoxon' or 't_test'",
    )
    parser.add_argument(
        "--min_standard_mean_deviation",
        type=float,
        default=0.5,
        help="Minimum standard mean deviation for DE gene filtering",
    )

    args = parser.parse_args()
    """
    notebook_pred_log_fc_dict, notebook_true_log_fc_dict = run_notebook_code(args)


    with open("notebook_pred_log_fc_dict.json", "w") as f:
        json.dump(dict_numpy_to_list(notebook_pred_log_fc_dict), f)
    with open("notebook_true_log_fc_dict.json", "w") as f:
        json.dump(dict_numpy_to_list(notebook_true_log_fc_dict), f)
    """
    result = run_new_code(args)
    breakpoint()
    # Similarly, save result.pred_log_fc_dict and result.true_log_fc_dict
    with open("new_pred_log_fc_dict.json", "w") as f:
        json.dump(dict_numpy_to_list(result.pred_log_fc_dict), f)
    with open("new_true_log_fc_dict.json", "w") as f:
        json.dump(dict_numpy_to_list(result.true_log_fc_dict), f)
