import logging
import sys
import anndata as ad
import pandas as pd
from pathlib import Path
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
        "--de_results_path",
        type=str,
        default="k562_data/wilcoxon_de_results.csv",
        help="Path to wilcoxon DE results CSV",
    )
    parser.add_argument(
        "--masked_h5ad_path",
        type=str,
        default="k562_data/zero_shot_0.5_de_genes_masked.h5ad",
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

    args = parser.parse_args()

    de_results_path = Path(args.de_results_path)
    masked_h5ad_path = Path(args.masked_h5ad_path)

    de_res_wilcoxon_df = pd.read_csv(de_results_path)[
        ["logfoldchanges", "pvals_adj", "target_gene", "names"]
    ]
    de_res_wilcoxon_df = de_res_wilcoxon_df[
        np.abs(de_res_wilcoxon_df["logfoldchanges"]) >= args.min_logfoldchanges
    ]
    de_res_wilcoxon_df = de_res_wilcoxon_df[
        de_res_wilcoxon_df["pvals_adj"] < args.pval_threshold
    ]
    de_res_wilcoxon_df = de_res_wilcoxon_df[["logfoldchanges", "target_gene", "names"]]

    masked_h5ad_df = ad.read_h5ad(masked_h5ad_path).obs[["gene"]]

    pred = np.load(str(args.predictions_path))
    sample_id = np.load(str(args.sample_id_path))
    target_genes = np.load(str(args.target_genes_path))

    pred_df = pd.DataFrame(
        {
            "sample_id": sample_id,
            "pred": list(pred),
            "target_genes": list(target_genes),
        }
    )

    task = PerturbationExpressionPredictionTask(min_de_genes=args.min_de_genes)
    task_input = PerturbationExpressionPredictionTaskInput(
        de_res_wilcoxon_df=de_res_wilcoxon_df, pred_df=pred_df
    )
    task.run(masked_h5ad_df, task_input)
