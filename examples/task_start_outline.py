import logging
import sys
import anndata as ad
import pandas as pd
from czbenchmarks.datasets.utils import load_dataset
from czbenchmarks.datasets.single_cell_labeled import SingleCellLabeledDataset
from pathlib import Path
from czbenchmarks.tasks.single_cell import (
    K562PerturbationTask,
)
import numpy as np
from czbenchmarks.tasks.types import CellRepresentation

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout) 
    #Hard coded filtering parameters for now:
    min_logfoldchanges = 1.0
    pval_threshold = 1e-4
    
    #Side load the necessary datasets for now!

    pred_dir = Path("/home/pbinder")
    de_results_path = Path(f"{pred_dir}/wilcoxon_de_results.csv")
    masked_h5ad_path = Path(f"{pred_dir}/zero_shot_0.5_de_genes_masked.h5ad")
    #masked_json_path = Path("/home/pbinder/zero_shot_0.5_de_genes_masked.json")
    
    #TODO: put these into the correct format for the task
    de_res_wilcoxon_df = pd.read_csv(de_results_path)[["logfoldchanges", "pvals_adj", "target_gene", "names"]]
    de_res_wilcoxon_df =  de_res_wilcoxon_df[np.abs(de_res_wilcoxon_df["logfoldchanges"]) >= min_logfoldchanges]
    de_res_wilcoxon_df = de_res_wilcoxon_df[de_res_wilcoxon_df["pvals_adj"] < pval_threshold] 
    de_res_wilcoxon_df = de_res_wilcoxon_df[["logfoldchanges", "target_gene", "names"]]

    masked_h5ad_df = ad.read_h5ad(masked_h5ad_path).obs[["gene"]]
    
    pred = np.load(f"{pred_dir}/predictions_merged.npy")
    sample_id = np.load(f"{pred_dir}/sample_id_merged.npy")
    target_genes = np.load(f"{pred_dir}/target_genes_merged.npy")

    #TODO: make this similar to the perturbation task input
    pred_df = pd.DataFrame({
        'sample_id': sample_id,
        'pred': list(pred),
        'target_genes': list(target_genes)
    })
    task = K562PerturbationTask(min_de_genes=20)
    #Question: what goes under run_task vs. compute metrics? 
    task.run(masked_h5ad_df, [de_res_wilcoxon_df, pred_df])
