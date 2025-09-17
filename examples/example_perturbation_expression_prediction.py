"""
Example script for running the Perturbation Expression Prediction Task.

CELL REPRESENTATION DATA (OPTIONAL):
By default, this script generates random cell representation data for testing.
You can provide your own cell representation data as an AnnData file:

For model_data_file (OPTIONAL):
    # Create an AnnData object with your cell representations
    import anndata as ad
    import numpy as np

    # Your cell representation matrix (cells x genes)
    cell_representations = np.random.rand(1000, 500)  # Replace with your actual data

    # Gene names should be in .var.index
    gene_names = ['GENE1', 'GENE2', 'GENE3', ...]  # Your gene identifiers

    # Cell/perturbation identifiers should be in .obs.index
    cell_ids = ['CELL_1', 'CELL_2', 'CELL_3', ...]  # Your cell identifiers

    # Create and save AnnData
    adata = ad.AnnData(X=cell_representations)
    adata.var.index = gene_names
    adata.obs.index = cell_ids
    adata.write_h5ad('my_model_data.h5ad')

    # Then run the script with:
    # python example_perturbation_expression_prediction.py --model_data_file my_model_data.h5ad

IMPORTANT:
- If no model_data_file is provided, random data will be generated for testing
- The gene ordering (.var.index) and cell ordering (.obs.index) from your file will be used
- Your data dimensions must match the dataset dimensions
"""

import logging
import sys
import argparse
import tempfile
import yaml
from pathlib import Path

import anndata as ad
import numpy as np

from czbenchmarks.datasets import load_dataset, SingleCellPerturbationDataset
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
    load_perturbation_task_input_from_saved_files,
)
from czbenchmarks.tasks.utils import print_metrics_summary
from czbenchmarks.tasks.types import CellRepresentation

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
        "--model_data_file",
        help="[OPTIONAL] Path to AnnData file (.h5ad) containing your cell representation data. "
        "If not provided, random data will be generated for testing. "
        "The file should have: cell representations in .X, gene names in .var.index, "
        "and cell identifiers in .obs.index. "
        "Example: adata.write_h5ad('my_model_data.h5ad')",
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

    # Choose approach based on flag
    if args.save_inputs:
        print("Using save/load approach...")
        # Save and load dataset task inputs
        task_inputs_dir = dataset.store_task_inputs()
        print(f"Task inputs saved to: {task_inputs_dir}")
        task_input = load_perturbation_task_input_from_saved_files(task_inputs_dir)
        print("Task inputs loaded from saved files")
    else:
        print("Creating task input directly from dataset...")
        # Create task input directly from dataset with separate fields
        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.control_matched_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
        )

    # Load model data or generate random data
    if args.model_data_file:
        model_adata = ad.read_h5ad(args.model_data_file)
        # Validate dimensions
        assert model_adata.shape == dataset.adata.shape, (
            f"Model data shape {model_adata.shape} does not match dataset shape {dataset.adata.shape}"
        )

        # Use the cell representation data from the file
        model_output: CellRepresentation = model_adata.X

        # Apply the gene and cell ordering from the model data to the task input
        task_input.adata.var.index = model_adata.var.index
        task_input.adata.uns["cell_barcode_index"] = model_adata.obs.index.astype(
            str
        ).values

    else:
        print("No model data file provided - generating random data for testing")

        # Generate random model output for testing
        model_output: CellRepresentation = np.random.rand(
            dataset.adata.shape[0], dataset.adata.shape[1]
        )

    # Run task
    task = PerturbationExpressionPredictionTask(metric=args.metric)
    metrics_dict = task.run(model_output, task_input)
    print_metrics_summary(metrics_dict)
