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


def generate_random_model_predictions(n_cells, n_genes):
    """This demonstrates the expected format for the model predictions.
    This should be an anndata file where the obs.index contains the cell
    barcodes and the var.index contains the genes. These should be the same or a
    subset of the genes and cells in the dataset. The X matrix should be the
    model predictions.
    """

    model_predictions: CellRepresentation = np.random.rand(n_cells, n_genes)
    # Put the predictions in an anndata object
    model_adata = ad.AnnData(X=model_predictions)

    # The same genes and cells (or a subset of them) should be in the model adata.
    model_adata.obs.index = (
        dataset.adata.obs.index.to_series().sample(frac=1, random_state=42).values
    )
    model_adata.var.index = (
        dataset.adata.var.index.to_series().sample(frac=1, random_state=42).values
    )
    return model_adata


if __name__ == "__main__":
    """Runs a task to calculate perturbation metrics. 

    As input, this uses a SingleCellPerturbationDataset. Currently, this assumes 
    data from the Replogle et al. 2022 dataset. Addtionally, this contains 
    differentially expressed genes for each perturbation. The extent of the 
    perturbation is merged with the wilcoxon test.
    
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
        help="Metric to use for DE analysis (wilcoxon only)",
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

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    # Instantiate a config and load the input data
    cfg = {
        "datasets": {
            "replogle_k562_essential_perturbpredict": {
                "percent_genes_to_mask": args.percent_genes_to_mask,
                "min_logfoldchange": args.min_logfoldchange,
                "pval_threshold": args.pval_threshold,
                "min_de_genes_to_mask": args.min_de_genes_to_mask,
            }
        }
    }

    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        dataset: SingleCellPerturbationDataset = load_dataset(
            "replogle_k562_essential_perturbpredict", config_path=str(cfg_path)
        )

    # This generates a sample model anndata file. In applications, this should be
    # provided by the user.
    model_adata = generate_random_model_predictions(
        dataset.adata.shape[0], dataset.adata.shape[1]
    )

    # Choose approach based on flag
    if args.save_inputs:
        logger.info("Using save/load approach...")
        # Save and load dataset task inputs
        task_inputs_dir = dataset.store_task_inputs()
        logger.info(f"Task inputs saved to: {task_inputs_dir}")
        task_input = load_perturbation_task_input_from_saved_files(task_inputs_dir)
        logger.info("Task inputs loaded from saved files")

        # Update with the model ordering of the genes and of the cells
        task_input.gene_index = model_adata.var.index
        task_input.cell_index = model_adata.obs.index
    else:
        logger.info("Creating task input directly from dataset...")
        # Create task input directly from dataset with separate fields
        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.control_matched_adata,
            target_conditions_dict=dataset.target_conditions_dict,
            de_results=dataset.de_results,
            gene_index=model_adata.var.index,
            cell_index=model_adata.obs.index,
        )
    # Convert model adata to cell representation
    model_output = model_adata.X
    # Run task
    task = PerturbationExpressionPredictionTask()
    metrics_dict = task.run(model_output, task_input)
    print_metrics_summary(metrics_dict)
