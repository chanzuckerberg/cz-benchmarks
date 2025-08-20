# This tests that the perturbation prediection task with the new data formats matches
# what is outputted in Maria's intial notebook.

import argparse
import os
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
)
from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
    load_perturbation_task_input_from_saved_files,
)

import json
from pathlib import Path
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


def run_notebook_code(args, sample_ids, target_genes, predictions):
    print("Loading notebook files...")
    # Read in the files that were saved in another file

    # Load notebook outputs from the specified directory
    nb_dir = args.notebook_task_inputs_path
    masked_notebook_adata = ad.read_h5ad(
        nb_dir / "notebook_adata_masked.h5ad", backed="r"
    )

    print("Notebook files loaded. Reading and filtering DE results...")
    # Read in and filter the DE results
    de_results = pd.read_csv(args.de_results_path)

    if args.metric_type == "wilcoxon":
        de_results = de_results[
            np.abs(de_results["logfoldchanges"]) >= args.min_logfoldchanges
        ]
        de_results = de_results[de_results["pvals_adj"] < args.pval_threshold]
    elif args.metric_type == "t-test":
        de_results = de_results[de_results["pval_adj"] < args.pval_threshold]
        de_results = de_results[
            de_results["smd"].abs() >= args.min_standard_mean_deviation
        ]
        de_results["target_gene"] = de_results["condition"]
        de_results["ensembl_id"] = de_results["gene"]
        de_results["names"] = de_results["gene"]

    print("DE results filtered. Loading predictions and grouping by sample...")
    # The original code is per model name. But here, we just have one model.
    predictions_dict = {}
    pred_log_fc_dict = {}
    true_log_fc_dict = {}
    metrics_dict = {
        "accuracy": {},
        "precision": {},
        "recall": {},
        "f1": {},
        "correlation": {},
    }
    df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "pred": list(predictions),
            "target_genes": list(target_genes),
        }
    )

    # Group predictions by sample
    print("Grouping predictions by sample_id...")
    genes_dict = {}
    for sample, group in tqdm(df.groupby("sample_id")):
        predictions_dict.setdefault(sample, []).append(group["pred"].to_numpy())
        if sample not in genes_dict:
            genes_dict[sample] = group["target_genes"].to_numpy()

    print("Finished grouping predictions. Iterating over all conditions...")
    # Iterate over all conditions
    for condition in masked_notebook_adata.obs["gene"].unique():
        # Select appropriate DE results. For model predicting log1p counts/binary class probabilities we use wilcoxon test results and logFC.
        # For model predicting z-scale expression we use t-test results and SMD.

        if condition == "non-targeting":
            continue
        condition_de_df = de_results[de_results["target_gene"] == condition]
        if len(condition_de_df) < args.min_de_genes:
            print(f"Not enough DE genes for condition {condition}. Skipping...")
            continue

        adata_condition = masked_notebook_adata[
            masked_notebook_adata.obs.index.str.endswith(f"_{condition}")
        ]
        condition_cells = adata_condition.obs[
            adata_condition.obs["gene"] != "non-targeting"
        ].index
        control_cells = adata_condition.obs[
            adata_condition.obs["gene"] == "non-targeting"
        ].index
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
            masked_genes = np.asarray(genes_dict[control_cells[0]])
            mask = masked_genes != "A"
            masked_genes = masked_genes[mask]

            n_masked_genes = len(masked_genes)
            if n_masked_genes == 0:
                print(f"No masked genes for condition {condition}. Skipping...")
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
            pred_log_fc = condition_predictions.mean(axis=0) - control_predictions.mean(
                axis=0
            )
            # Select column for true log fold change
            col = "smd" if args.metric_type == "t-test" else "logfoldchanges"
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
    print("Finished processing all conditions. Calculating metrics...")
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
    print("All metrics calculated for notebook code.")
    return pred_log_fc_dict, true_log_fc_dict, metrics_dict


def run_new_code(model_output, args):
    print("Loading dataset from task inputs...")
    task_input = load_perturbation_task_input_from_saved_files(args.new_saved_dir)
    print("Done loading dataset from task inputs.")
    task = PerturbationExpressionPredictionTask(metric=args.metric_type)
    print("Running task.run()...")
    result = task._run_task(model_output, task_input)
    print("Running task._compute_metrics()...")
    # Run the metrics from the task
    metric_results = task._compute_metrics(task_input, result)

    print("New code run complete.")
    return result, metric_results


def generate_model_predictions(masked_notebook_adata, args):
    """
    Generate a model_output matrix for the given masked_notebook_adata using the provided
    target_genes, sample_id, and predictions files.

    Returns:
        model_output: np.ndarray of shape (n_cells, n_genes)
    """
    with (args.notebook_task_inputs_path / "notebook_target_genes_to_save.json").open(
        "r"
    ) as f:
        target_genes_to_save = json.load(f)
    print("Generating model predictions matrix...")
    model_output: CellRepresentation = np.random.rand(
        masked_notebook_adata.shape[0], masked_notebook_adata.shape[1]
    )
    obs_index = masked_notebook_adata.obs.index

    # Speed up by using numpy and pandas vectorized lookups instead of repeated .index() calls
    row_index = np.array([i.split("_")[0] for i in obs_index])
    col_index = np.array(masked_notebook_adata.var.index)
    row_lookup = {barcode: idx for idx, barcode in enumerate(row_index)}
    col_lookup = {gene: idx for idx, gene in enumerate(col_index)}
    sample_ids = []
    target_genes = []
    predictions = []
    all_row_indices = []
    all_col_indices = []
    for barcode, genes in target_genes_to_save.items():
        row_idx = row_lookup.get(barcode.split("_")[0])
        if row_idx is None:
            continue
        mask = np.array([gene in col_lookup for gene in genes])
        filtered_genes = np.array(genes)[mask]
        col_indices = [col_lookup[gene] for gene in filtered_genes]
        sample_ids.extend([barcode] * len(filtered_genes))
        target_genes.extend(filtered_genes)
        predictions.extend(model_output[row_idx, col_indices])
        all_row_indices.extend([row_idx] * len(filtered_genes))
        all_col_indices.extend(col_indices)

    print("Model predictions matrix generated.")
    return model_output, sample_ids, target_genes, predictions


if __name__ == "__main__":
    print("Parsing arguments...")
    parser = argparse.ArgumentParser(
        description="Run K562 Perturbation Task with custom file paths and filtering parameters."
    )
    parser.add_argument(
        "--metric_type",
        type=str,
        choices=["wilcoxon", "t-test"],
        default="wilcoxon",
        help="Metric type to use for DE gene filtering: 'wilcoxon' or 't_test'",
    )
    parser.add_argument(
        "--percent_genes_to_mask",
        type=float,
        default=1.0,
        help="Percentage of genes to mask, should be 0.5 or 1.0",
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
        default=5,
        help="Minimum number of DE genes for perturbation condition",
    )
    parser.add_argument(
        "--de_results_path",
        type=str,
        default="data2/czbenchmarks/replogle2022/K562/zero_shot_benchmark/{metric_type}/de_results.csv",
        help="Path to de_results .csv file",
    )

    parser.add_argument(
        "--initial_data_set_path",
        type=str,
        default=f"{os.environ['HOME']}/.cz-benchmarks/datasets/replogle_k562_essential_perturbpredict_de_results_control_cells.h5ad",
        help="Path to target_genes .npy file",
    )

    parser.add_argument(
        "--new_saved_dir",
        type=str,
        default=os.environ["HOME"]
        + "/.cz-benchmarks/datasets/replogle_k562_essential_perturbpredict_de_results_control_cells_task_inputs/single_cell_perturbation_{metric_type}_{percent_genes_to_mask}",
    )

    parser.add_argument(
        "--notebook_task_inputs_path",
        type=str,
        default="notebook_task_inputs_{metric_type}_{percent_genes_to_mask}",
    )

    args = parser.parse_args()
    mask_portion = (
        str(args.percent_genes_to_mask) if args.percent_genes_to_mask == 0.5 else "all"
    )
    args.de_results_path = args.de_results_path.format(metric_type=args.metric_type)
    args.notebook_task_inputs_path = Path(
        args.notebook_task_inputs_path.format(
            metric_type=args.metric_type,
            percent_genes_to_mask=args.percent_genes_to_mask,
        )
    )
    args.new_saved_dir = Path(
        args.new_saved_dir.format(
            metric_type=args.metric_type,
            percent_genes_to_mask=args.percent_genes_to_mask,
        )
    )

    print("Loading initial dataset...")
    initial_datast = ad.read_h5ad(args.initial_data_set_path, backed="r")

    print("Generating model_output...")
    model_output, sample_ids, target_genes, predictions = generate_model_predictions(
        initial_datast, args
    )
    print("Running new code for new_result and new_metrics...")
    new_result, new_metrics = run_new_code(model_output, args)
    print("Running notebook code for comparison...")

    notebook_pred_log_fc_dict, notebook_true_log_fc_dict, metrics_dict = (
        run_notebook_code(args, sample_ids, target_genes, predictions)
    )

    print("Processing new_metrics into dictionary...")
    new_metrics_dict = {}
    for metric in new_metrics:
        mapping = {}
        for ent in new_metrics[metric]:
            mapping[ent.params["condition"]] = ent.value
        new_metrics_dict[metric] = mapping

    # Make sure all keys in new are in notebook, and compare notebook to new
    with (args.notebook_task_inputs_path / "gene_map.json").open("r") as f:
        notebook_gene_map = json.load(f)
    for k in new_result.true_log_fc_dict:
        assert notebook_gene_map[k] in notebook_true_log_fc_dict, (
            f"Key {k} missing in notebook_true_log_fc_dict"
        )
        assert np.allclose(
            np.sort(notebook_true_log_fc_dict[notebook_gene_map[k]][0]),
            np.sort(new_result.true_log_fc_dict[k]),
        )
    for k in new_result.pred_log_fc_dict:
        assert notebook_gene_map[k] in notebook_pred_log_fc_dict, (
            f"Key {k} missing in notebook_pred_log_fc_dict"
        )
        assert np.allclose(
            np.sort(notebook_pred_log_fc_dict[notebook_gene_map[k]][0]),
            np.sort(new_result.pred_log_fc_dict[k]),
        )

    print("Checking equivalency of metrics keys...")
    assert new_metrics_dict.keys() == metrics_dict.keys()
    for metric in new_metrics_dict:
        for condition in new_metrics_dict[metric]:
            assert np.isclose(
                new_metrics_dict[metric][condition],
                metrics_dict[metric][notebook_gene_map[condition]],
            )
    print("All checks passed. Test complete.")
