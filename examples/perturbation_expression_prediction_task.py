import logging
import sys
from czbenchmarks.datasets import load_dataset
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
import numpy as np
import argparse
from czbenchmarks.datasets import SingleCellPerturbationDataset
from czbenchmarks.tasks.types import CellRepresentation

if __name__ == "__main__":
    """Runs a task to calculate perturbation metrics. 
    Assumes wilcoxon DE results and a masked h5ad file."""
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)

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
        "--h5ad_path",
        type=str,
        default="new_data/replogle_k562_essential_perturbpredict.h5ad",
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
        "--metric_type",
        type=str,
        choices=["wilcoxon", "t_test"],
        default="wilcoxon",
        help="Metric type to use for DE gene filtering: 'wilcoxon' or 't_test'",
    )

    # Load the input data
    args = parser.parse_args()
    dataset: SingleCellPerturbationDataset = load_dataset(
        "replogle_k562_essential_perturbpredict"
    )
    model_output: CellRepresentation = np.random.rand(
        dataset.adata.shape[0], dataset.adata.shape[1]
    )
    np.save("/tmp/random_model_output.npy", model_output)
    task = PerturbationExpressionPredictionTask()

    task = PerturbationExpressionPredictionTask()
    task_input = PerturbationExpressionPredictionTaskInput(
        de_results=dataset.de_results,
        var_index=dataset.control_matched_adata.var.index,
        masked_adata_obs=dataset.control_matched_adata.obs,
        target_conditions_to_save=dataset.target_conditions_to_save,
    )
    task.run(model_output, task_input)
