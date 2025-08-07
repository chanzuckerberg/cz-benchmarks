import logging
import sys
import anndata as ad
import pandas as pd
from czbenchmarks.tasks.single_cell import (
    PerturbationExpressionPredictionTask,
    PerturbationExpressionPredictionTaskInput,
)
import numpy as np
import argparse

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

    # Load the input data
    args = parser.parse_args()

    adata = ad.read_h5ad(args.h5ad_path, backed="r")

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

    task = PerturbationExpressionPredictionTask(min_de_genes=args.min_de_genes)
    task_input = PerturbationExpressionPredictionTaskInput(
        de_results=adata.uns["de_results_wilcoxon"],
        control_cells_ids=adata.uns["control_cells_ids"],
    )
    task.run(pred_df, task_input)
