import czbenchmarks.datasets.utils_single_cell as scp

import os
import anndata as ad
import time
import logging
import json
import anndata as an
# import scanpy as sc
# import pandas as pd
import numpy as np
# from pandas.testing import assert_frame_equal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    filtered_data_path = f"{os.environ['HOME']}/.cz-benchmarks/datasets/replogle_k562_essential_perturbpredict_de_results_control_cells.h5ad"
    adata_filtered = ad.read_h5ad(filtered_data_path, backed=None)
    
    #########
    cr4_data_path = (
    "/data2/czbenchmarks/replogle2022/raw_h5ad_from_cr4/K562_essential_mtx.h5ad"
    )
    adata_cr4 = ad.read_h5ad(cr4_data_path, backed=None)
    adata_cr4.var.index.name = "gene_id"
    adata_cr4.var.rename(columns={"gene_name": "gene"}, inplace=True)
    adata_cr4.obs.rename(columns={"gene_id": "condition"}, inplace=True)
    
    deg_name = "wilcoxon"
    with open(f"/data2/unique_conditions_{deg_name}.json", 'r') as fh:
        unique_conditions = json.load(fh)
    ########

    # de_results_wilcoxon = pd.DataFrame(adata_filtered.uns["de_results_wilcoxon"])
    # de_results_ttest = pd.DataFrame(adata_filtered.uns["de_results_t_test"])
    control_cells_ids = adata_filtered.uns["control_cells_ids"]
    control_cells_ids.pop("non-targeting", None)

    #########
    mask = adata_cr4.obs.condition.isin(unique_conditions)
    shared_conditions = adata_cr4.obs[~mask].condition.unique().tolist()
    shared_conditions = [c for c in shared_conditions if c != 'non-targeting']
    # adata_cr4 = an.concat([adata_cr4[mask], adata_cr4[~mask]])
    new_control_cells_ids = {
        k: control_cells_ids[k] for k in [shared_conditions[0]] + unique_conditions + shared_conditions[1:]
        }
    #########

    # num_conditions = 10
    # condition_list = np.asarray(list(control_cells_ids.keys()))
    # condition_list = np.random.choice(condition_list, size=num_conditions, replace=False)
    # condition_list = list(condition_list)
    # new_control_cells_ids = {k: control_cells_ids[k] for k in condition_list}
    # control_cells_ids = new_control_cells_ids

    # for deg_test_name in ["wilcoxon", "t-test"]:
    for deg_test_name in [deg_name]:
        if deg_test_name == "t-test":
            z_scale_group_col = "gem_group"
        else:
            z_scale_group_col = None

        print(f"Running {deg_test_name} DE analysis")
        start_time = time.time()
        results, skip_conditions = scp.run_multicondition_dge_analysis(
            adata=adata_cr4, #adata_filtered,
            condition_key="condition",
            control_name="non-targeting",
            control_cells_ids=new_control_cells_ids, #control_cells_ids,
            deg_test_name=deg_test_name,
            filter_min_cells=10, #0,
            filter_min_genes=10, #0,
            min_pert_cells=1, #1,
            remove_avg_zeros=False,
            return_merged_adata=False,
            z_scale_group_col=z_scale_group_col,
        )
        end_time = time.time()
        print(f"Time taken for {len(control_cells_ids)} conditions: {end_time - start_time} seconds")

        if results.shape[0] != (adata_filtered.obs.condition.nunique() - 1) * adata_filtered.n_vars:
            print("Warning: Unexpected number of results returned")
        
        start_time = time.time()
        results.to_parquet(f"/data2/czbenchmarks/replogle_k562_essentials_de_results_{deg_test_name}.arrow")
        
        end_time = time.time()
        print(f"Time taken to save results: {end_time - start_time} seconds")
