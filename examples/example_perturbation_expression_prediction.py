import logging
import sys
import argparse
from czbenchmarks.datasets import load_dataset
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
    load_perturbation_task_input_from_saved_files,
)
from czbenchmarks.tasks.utils import print_metrics_summary
import numpy as np
import pandas as pd
from czbenchmarks.datasets import SingleCellPerturbationDataset
from czbenchmarks.tasks.types import CellRepresentation
import tempfile
import yaml
from pathlib import Path

if __name__ == "__main__":
    """Runs a task to calculate perturbation metrics. 

    As input, this uses a SingleCellPerturbationDataset. Currently, this assumes 
    data from the Replogle et al. 2022 dataset. Addtionally, this contains 
    differentially expressed genes for each perturbation. The extent of the 
    perturbation is merged with the willcoxon test or t-test.
    
    The dataset is filtered based on the type of statistical test, along with the minimum 
    number of differentially expressed genes, maximum p-value, and the minimum 
    log fold change or standardized mean difference. During the dataset generation, 
    the specified percentage of genes to mask is randomly selected.
    
    The dataset can be saved after filtering, and then loaded back in.
    
    In this example, a random model output is used. Instead, any model output of
    the same shape as the dataset's adata can be used.
    
    The task computes the log fold change in model predicted expression of 
    differentially expressed genes between perturbed and non-targeting groups.
    It then calculates the correlation between ground truth and predicted log 
    fold change for each condition using a variety of metrics.    
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run perturbation expression prediction task"
    )
    parser.add_argument(
        "--save-inputs",
        action="store_true",
        help="Save dataset task inputs to disk and load them back (demonstrates save/load functionality)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="wilcoxon",  # Set this to correspond to the type of statistical test used to determine differentially expressed genes
        help="Metric to use for DE analysis",
    )
    parser.add_argument(
        "--percent_genes_to_mask",
        type=float,
        default=1.0,
        help="Percentage of genes to mask",
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
        "--gene_index_file",
        help="File with gene names (one per line) to use as gene_index",
    )
    parser.add_argument(
        "--perturb_index_file",
        help="File with perturbation names (one per line) to use as perturb_index",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # Instantiate a config and load the input data
    cfg = {
        "datasets": {
            "replogle_k562_essential_perturbpredict": {
                "percent_genes_to_mask": args.percent_genes_to_mask,
                "deg_test_name": args.metric,
                "min_logfoldchange": args.min_logfoldchange,
                "pval_threshold": args.pval_threshold,
                "min_de_genes_to_mask": args.min_de_genes_to_mask,
                "min_smd": args.min_smd,
            }
        }
    }

    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        dataset: SingleCellPerturbationDataset = load_dataset(
            "replogle_k562_essential_perturbpredict", config_path=str(cfg_path)
        )  # TODO: Once PR 381 is merged, use the new load_local_dataset function

    # Load custom indices if provided, with assertions
    gene_index = dataset.control_matched_adata.var.index
    if args.gene_index_file:
        gene_index = pd.Index(open(args.gene_index_file).read().strip().split("\n"))
        assert len(gene_index) == len(dataset.control_matched_adata.var.index), (
            f"Length of gene_index from file ({len(gene_index)}) does not match "
            f"length of dataset.control_matched_adata.var.index ({len(dataset.control_matched_adata.var.index)})"
        )

    perturb_index = dataset.adata.obs.index
    if args.perturb_index_file:
        perturb_index = pd.Index(
            open(args.perturb_index_file).read().strip().split("\n")
        )
        assert len(perturb_index) == len(dataset.adata.obs.index), (
            f"Length of perturb_index from file ({len(perturb_index)}) does not match "
            f"length of dataset.adata.obs.index ({len(dataset.adata.obs.index)})"
        )

    # Choose approach based on flag
    if args.save_inputs:
        print("Using save/load approach...")
        # Save and load dataset task inputs
        task_inputs_file = dataset.store_task_inputs()
        print(f"Task inputs saved to: {task_inputs_file}")
        task_input = load_perturbation_task_input_from_saved_files(task_inputs_file)
        # Override with custom indices if provided
        task_input.gene_index = gene_index
        task_input.perturb_index = perturb_index
        print("Task inputs loaded from saved files")
    else:
        print("Creating task input directly from dataset...")
        # Create task input directly from dataset
        task_input = PerturbationExpressionPredictionTaskInput(
            de_results=dataset.de_results,
            gene_index=gene_index,
            masked_adata_obs=dataset.control_matched_adata.obs,
            target_conditions_dict=dataset.target_conditions_dict,
            perturb_index=perturb_index,
        )

    # Generate random model output
    model_output: CellRepresentation = np.random.rand(
        dataset.adata.shape[0], dataset.adata.shape[1]
    )

    # Run task
    task = PerturbationExpressionPredictionTask(metric=args.metric)
    metrics_dict = task.run(model_output, task_input)
    print_metrics_summary(metrics_dict)
