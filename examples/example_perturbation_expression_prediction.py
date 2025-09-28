import logging
import sys
import argparse
from typing import Optional
import omegaconf
import hydra
from hydra.utils import instantiate
import numpy as np
import anndata as ad
import scanpy as sc

from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
from czbenchmarks.tasks.single_cell.perturbation_expression_prediction import (
    load_perturbation_task_input_from_saved_files,
)
from czbenchmarks.tasks.utils import print_metrics_summary
from czbenchmarks.datasets import SingleCellPerturbationDataset
from czbenchmarks.tasks.types import CellRepresentation

from czbenchmarks.utils import initialize_hydra
from czbenchmarks.file_utils import download_file_from_remote

logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run perturbation expression prediction task"
    )
    parser.add_argument(
        "--save-task-inputs",
        action="store_true",
        help="Save dataset task inputs to disk and load them back "
        "(demonstrates save/load functionality)",
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
        help="Minimum absolute log-fold change for DE filtering",
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
    return parser.parse_args()


def load_dataset_config(
    dataset_name: str,
    config_name: str = "datasets",
    dataset_update_dict: Optional[dict] = None,
):
    """Customize dataset class instantiation parameters using cli args

    Args:
        dataset_name: Name of the dataset to load
        dataset_update_dict: Optional dictionary of dataset parameters to update

    Returns:
        Dataset configuration
    """
    initialize_hydra()
    cfg = hydra.compose(config_name=config_name)
    dataset_cfg = cfg.datasets[dataset_name]
    if dataset_update_dict:
        with omegaconf.open_dict(dataset_cfg) as d:
            d.update(dataset_update_dict)

    return dataset_cfg


def generate_random_model_predictions(n_cells, n_genes):
    """This demonstrates the expected format for the model predictions.
    This should be an anndata file where the obs.index contains the cell
    barcodes and the var.index contains the genes. These should be the same or a
    subset of the genes and cells in the dataset. The X matrix should be the
    model predictions.
    """

    model_predictions: CellRepresentation = np.random.randint(
        0, 100, (n_cells, n_genes)
    ).astype(float)
    # Put the predictions in an anndata object
    model_adata = ad.AnnData(X=model_predictions)

    # The same genes and cells (or a subset of them) should be in the model adata.
    model_adata.obs.index = (
        dataset.adata.obs.index.to_series().sample(frac=1, random_state=42).values
    )
    model_adata.var.index = (
        dataset.adata.var.index.to_series().sample(frac=1, random_state=42).values
    )

    # Log normalize the data -- in real applications, additional preprocessing
    # Steps may be applied
    sc.pp.log1p(model_adata)

    return model_adata


if __name__ == "__main__":
    """Runs a task to calculate perturbation metrics. 

    As input, this uses a SingleCellPerturbationDataset with data from the
    Replogle et al. 2022 dataset. Additionally, this dataset contains 
    control-matched differentially expressed genes for each perturbation. 
    The extent of the perturbation is measured with the Wilcoxon rank-sum test.
    
    The dataset is filtered based on statistical parameters, along with the minimum 
    number of differentially expressed genes, and then treatment cells are
    matched with control cells. The differentially expressed genes are used in
    the selection of genes for masking. 
    
    There is an optinal flag, --save-inputs, to save the outputs of the dataset
    for loading into the task at a future time.
    
    In this example, a random model output is used. Instead, any model output of
    the same shape as the dataset's adata can be used.
    
    The task uses the pre-computed log fold change in model predicted expression of 
    genes between perturbed and control groups. It then calculates the correlation 
    between ground truth and predicted log fold change for each condition.
    """

    args = parse_args()
    dataset_name = "replogle_k562_essential_perturbpredict"
    logging.info(f"Loading dataset for {dataset_name} with args: {args}")

    dataset_update_dict = {
        "percent_genes_to_mask": args.percent_genes_to_mask,
        "min_logfoldchange": args.min_logfoldchange,
        "pval_threshold": args.pval_threshold,
        "min_de_genes_to_mask": args.min_de_genes_to_mask,
    }
    # TODO could be used to improve flexibility of current load_dataset function
    dataset_cfg = load_dataset_config(
        dataset_name=dataset_name, dataset_update_dict=dataset_update_dict
    )
    dataset_cfg["path"] = download_file_from_remote(dataset_cfg["path"])

    dataset: SingleCellPerturbationDataset = instantiate(dataset_cfg)

    # Load data -- the dataset class has an optional validation method that
    # can be run while loading and/or afterwards
    dataset.load_data()  # Add validate_input_data=True to validate while loading
    dataset.validate()

    # This generates a sample model anndata file. In applications,
    # this should contain the model predictions and should be provided by the user.
    model_adata = generate_random_model_predictions(
        dataset.adata.shape[0], dataset.adata.shape[1]
    )

    # Choose approach based on flag
    if args.save_task_inputs:
        logging.info("Using save/load approach...")

        # Save and load dataset task inputs -- useful for caching dataset output
        task_inputs_dir = dataset.store_task_inputs()
        logging.info(f"Task inputs saved to: {task_inputs_dir}")

        task_input = load_perturbation_task_input_from_saved_files(task_inputs_dir)
        logging.info("Task inputs loaded from saved files")

        # FIXME MICHELLE flow doesn't make sense yet
        # Update with the model ordering of the genes and of the cells
        task_input.gene_index = model_adata.var.index
        task_input.cell_index = model_adata.obs.index
    else:
        logging.info("Creating task input directly from dataset...")

        task_input = PerturbationExpressionPredictionTaskInput(
            adata=dataset.control_matched_adata,
            target_condition_dict=dataset.target_condition_dict,
            de_results=dataset.de_results,
            gene_index=model_adata.var.index,
            cell_index=model_adata.obs.index,
        )

    # Run task
    task = PerturbationExpressionPredictionTask(
        condition_key=dataset_cfg["condition_key"],
        control_name=dataset_cfg["control_name"],
        condition_control_sep=dataset_cfg["condition_control_sep"],
        de_gene_col=dataset_cfg["de_gene_col"],
    )

    # Convert model adata to cell representation
    model_output = model_adata.X
    metrics_dict = task.run(cell_representation=model_output, task_input=task_input)
    print_metrics_summary(metrics_dict)

    # FIXME MICHELLE this throws an error because of non-log-normalized data
    # Compute baselines
    baseline_model = task.compute_baseline(
        cell_representation=dataset.adata.X, baseline_type="median"
    )
    baseline_metrics_dict = task.run(
        cell_representation=baseline_model, task_input=task_input
    )
    logging.info("Baseline metrics:")
    print_metrics_summary(baseline_metrics_dict)
