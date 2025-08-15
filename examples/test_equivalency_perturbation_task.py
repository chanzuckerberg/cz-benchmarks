# This tests that the perturbation prediection task with the new data formats matches
# what is outputted in Maria's intial notebook.

import argparse
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
import json
import os
from pathlib import Path
from czbenchmarks.datasets import (
    SingleCellPerturbationDataset,
    load_dataset,
)
from czbenchmarks.tasks.types import CellRepresentation


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


def run_notebook_code(args, model_output):
    # Read in the files that were saved in another file

    # Load notebook outputs from the specified directory
    nb_dir = args.notebook_task_inputs_path
    masked_adata = ad.read_h5ad(str(nb_dir / "notebook_adata_masked.h5ad"))
    with (nb_dir / "notebook_target_genes_to_save.json").open("r") as f:
        target_genes_to_save = json.load(f)
    with (nb_dir / "gene_map.json").open("r") as f:
        gene_map = json.load(f)

    
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


    # The original code is per model name. But here, we just have one model.
    predictions_dict = {}
    genes_dict = {}
    pred_log_fc_dict = {}
    true_log_fc_dict = {}
    metrics_dict = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1": {},
        "correlation": {},
    }
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
            if n_masked_genes == 0:
                continue
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
    for condition in pred_log_fc_dict:
        pred_log_fc_all = np.concatenate(pred_log_fc_dict[condition])
        true_log_fc_all = np.concatenate(true_log_fc_dict[condition])
        ids = np.where(~np.isnan(pred_log_fc_all) & ~np.isinf(pred_log_fc_all))[0]

        # Binary up/down regulation
        pred_binary = (pred_log_fc_all[ids] > 0).astype(int)
        true_binary = (true_log_fc_all[ids] > 0).astype(int)

        # Metrics
        accuracy = accuracy_score(true_binary, pred_binary) if len(true_binary) else 0
        precision = (
            precision_score(true_binary, pred_binary, average="binary")
            if len(true_binary)
            else 0
        )
        recall = (
            recall_score(true_binary, pred_binary, average="binary")
            if len(true_binary)
            else 0
        )
        f1 = (
            f1_score(true_binary, pred_binary, average="binary")
            if len(true_binary)
            else 0
        )
        correlation = (
            spearmanr(pred_log_fc_all[ids], true_log_fc_all[ids])[0] if len(ids) else 0
        )
        if np.isnan(accuracy):
            accuracy = 0
        if np.isnan(correlation):
            correlation = 0

        metrics_dict["accuracy"][condition] = accuracy
        metrics_dict["precision"][condition] = precision
        metrics_dict["recall"][condition] = recall
        metrics_dict["f1"][condition] = f1
        metrics_dict["correlation"][condition] = correlation
    return pred_log_fc_dict, true_log_fc_dict, metrics_dict


def run_new_code(dataset, args, model_output):
    task = PerturbationExpressionPredictionTask()
    task_input = PerturbationExpressionPredictionTaskInput(
        de_results=dataset.de_results,
        var_index=dataset.adata.var.index,
        masked_adata_obs=dataset.control_matched_adata.obs,
        target_genes_to_save=dataset.target_genes_to_save,
    )
    # Fix the undefined pred_df variable - we need to create it from model_output
    pred_df = pd.DataFrame(model_output, index=dataset.adata.obs.index, columns=dataset.adata.var.index)
    result = task._run_task(pred_df, task_input)
    # Run the metrics from the task
    metric_results = task._compute_metrics(task_input, result)

    return result, metric_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run K562 Perturbation Task with custom file paths and filtering parameters."
    )
    parser.add_argument(
        "--metric_type",
        type=str,
        choices=["wilcoxon", "t_test"],
        default="wilcoxon",
        help="Metric type to use for DE gene filtering: 'wilcoxon' or 't_test'",
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
        "--min_standard_mean_deviation",
        type=float,
        default=0.5,
        help="Minimum standard mean deviation for DE gene filtering",
    )
    parser.add_argument(
        "--min_de_genes",
        type=int,
        default=10,
        help="Minimum number of DE genes for perturbation condition",
    )
    parser.add_argument(
        "--new_dataset_path",
        type=str,
        default="datasets/replogle_k562_essential_perturbpredict.h5ad",
        help="Path to masked h5ad file",
    )

    parser.add_argument(
        "--de_results_path",
        type=str,
        default="replogle2022/K562/zero_shot_benchmark/{metric_type}/de_results.csv",
        help="Path to de_results .csv file",
    )

    parser.add_argument(
        "--masked_h5ad_path",
        type=str,
        default="replogle2022/K562/zero_shot_benchmark/{metric_type}/zero_shot_0.5_de_genes_masked.h5ad",
        help="Path to masked h5ad file",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="replogle2022/K562/sample_model_output/{metric_type}/target_genes_0.5_de_genes_masked/predictions_merged.npy",
        help="Path to predictions .npy file",
    )
    parser.add_argument(
        "--sample_id_path",
        type=str,
        default="replogle2022/K562/sample_model_output/{metric_type}/target_genes_0.5_de_genes_masked/sample_id_merged.npy",
        help="Path to sample_id .npy file",
    )
    parser.add_argument(
        "--target_genes_path",
        type=str,
        default="replogle2022/K562/sample_model_output/{metric_type}/target_genes_0.5_de_genes_masked/target_genes_merged.npy",
        help="Path to target_genes .npy file",
    )
    parser.add_argument("--notebook_task_inputs_path", type=Path, default=Path("notebook_task_inputs_20250814_144000"))

    args = parser.parse_args()

    # Format the paths with the metric type
    """
    args.masked_h5ad_path = args.masked_h5ad_path.format(metric_type=args.metric_type)
    args.predictions_path = args.predictions_path.format(metric_type=args.metric_type)
    args.sample_id_path = args.sample_id_path.format(metric_type=args.metric_type)
    args.target_genes_path = args.target_genes_path.format(metric_type=args.metric_type)
    args.de_results_path = args.de_results_path.format(metric_type=args.metric_type)
    """
    dataset: SingleCellPerturbationDataset = load_dataset(
        "replogle_k562_essential_perturbpredict"
    )
    model_output: CellRepresentation = np.random.rand(
        dataset.adata.shape[0], dataset.adata.shape[1]
    )

    new_result, new_metrics = run_new_code(dataset, args, model_output)

    notebook_pred_log_fc_dict, notebook_true_log_fc_dict, metrics_dict = (
        run_notebook_code(args, model_output)
    )

    new_metrics_dict = {}
    for metric in new_metrics:
        mapping = {}
        for ent in new_metrics[metric]:
            mapping[ent.params["condition"]] = ent.value
        new_metrics_dict[metric] = mapping
    assert len(notebook_pred_log_fc_dict) == len(new_result.pred_log_fc_dict)
    assert len(notebook_true_log_fc_dict) == len(new_result.true_log_fc_dict)

    for k in notebook_true_log_fc_dict:
        assert (
            np.array_equal(np.sort(new_result.true_log_fc_dict[k]), 
                          np.sort(notebook_true_log_fc_dict[k][0]))
        )
    for k in notebook_pred_log_fc_dict:
        assert (
            np.array_equal(np.sort(new_result.pred_log_fc_dict[k]), 
                          np.sort(notebook_pred_log_fc_dict[k][0]))
        )

    assert new_metrics_dict.keys() == metrics_dict.keys()
    for metric in new_metrics_dict:
        for condition in new_metrics_dict[metric]:
            assert (
                new_metrics_dict[metric][condition] != metrics_dict[metric][condition]
            )
