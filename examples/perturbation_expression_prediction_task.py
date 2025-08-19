import logging
import sys
import argparse
from czbenchmarks.datasets import load_dataset
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
import pandas as pd
import numpy as np
from czbenchmarks.datasets import SingleCellPerturbationDataset
from czbenchmarks.tasks.types import CellRepresentation
def print_metrics_summary(metrics_dict):
    """Print a nice summary table of all metrics."""
    
    # Extract all conditions from any metric type
    conditions = [result.params["condition"] for result in metrics_dict["accuracy"]]
    
    # Create a summary dictionary
    summary_data = []
    for condition in conditions:
        row = {"condition": condition}
        
        # Extract values for each metric type
        for metric_name, results in metrics_dict.items():
            # Find the result for this condition
            condition_result = next(r for r in results if r.params["condition"] == condition)
            row[metric_name] = f"{condition_result.value:.4f}"
        
        summary_data.append(row)
    
    # Create and print DataFrame
    df = pd.DataFrame(summary_data)
    print("\n=== Perturbation Expression Prediction Results ===")
    print(df.to_string(index=False))
    
    # Print overall statistics
    print(f"\nSummary Statistics across {len(conditions)} conditions:")
    for metric_name, results in metrics_dict.items():
        values = [r.value for r in results]
        print(f"{metric_name.title()}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

if __name__ == "__main__":
    """Runs a task to calculate perturbation metrics. 
    Assumes wilcoxon DE results and a masked h5ad file."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run perturbation expression prediction task"
    )
    parser.add_argument(
        "--save-inputs", 
        action="store_true",
        help="Save dataset task inputs to disk and load them back (demonstrates save/load functionality)"
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
        task_input = PerturbationExpressionPredictionTask.load_from_task_inputs(task_inputs_dir)
        print("Task inputs loaded from saved files")
    else:
        print("Creating task input directly from dataset...")
        # Create task input directly from dataset
        task_input = PerturbationExpressionPredictionTaskInput(
            de_results=dataset.de_results,
            var_index=dataset.control_matched_adata.var.index,
            masked_adata_obs=dataset.control_matched_adata.obs,
            target_conditions_to_save=dataset.target_conditions_to_save,
        )
    
    # Generate random model output
    model_output: CellRepresentation = np.random.rand(
        dataset.adata.shape[0], dataset.adata.shape[1]
    )
    np.save("/tmp/random_model_output.npy", model_output)

    # Run task
    task = PerturbationExpressionPredictionTask()
    metrics_dict = task.run(model_output, task_input)
    print_metrics_summary(metrics_dict)