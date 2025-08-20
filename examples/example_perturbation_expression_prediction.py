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
from czbenchmarks.datasets import SingleCellPerturbationDataset
from czbenchmarks.tasks.types import CellRepresentation

if __name__ == "__main__":
    """Runs a task to calculate perturbation metrics. 

    As input, this uses a SingleCellPerturbationDataset. Currently, this assumes 
    data from the Replogle et al. 2022 dataset. Addtionally, this contains 
    differentially expressed genes for each perturbation. The extent of the 
    perturbation is merged with the willcoxon test or t-test.
    
    The dataset is filtered based on the type of t-test, along with the minimum 
    number of differentially expressed genes, maximum p-value, and the minimum 
    log fold change or standardized mean difference.
    
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

    # Load the input data
    dataset: SingleCellPerturbationDataset = load_dataset(
        "replogle_k562_essential_perturbpredict"
    )

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
        # Create task input directly from dataset
        task_input = PerturbationExpressionPredictionTaskInput(
            de_results=dataset.de_results,
            var_index=dataset.control_matched_adata.var.index,
            masked_adata_obs=dataset.control_matched_adata.obs,
            target_conditions_to_save=dataset.target_conditions_to_save,
            row_index=dataset.adata.obs.index,
        )

    # Generate random model output
    model_output: CellRepresentation = np.random.rand(
        dataset.adata.shape[0], dataset.adata.shape[1]
    )

    # Run task
    task = PerturbationExpressionPredictionTask()
    metrics_dict = task.run(model_output, task_input)
    print_metrics_summary(metrics_dict)
