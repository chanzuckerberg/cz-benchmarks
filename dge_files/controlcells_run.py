import czbenchmarks.datasets.utils_single_cell as scp
from tqdm import tqdm
import os
import anndata as ad
import time
import logging
import json
import anndata as an

import scanpy as sc
import pandas as pd
import numpy as np
from czbenchmarks.datasets.utils_control_cells import get_matched_controls

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    filtered_data_path = f"{os.environ['HOME']}/.cz-benchmarks/datasets/replogle_k562_essential_perturbpredict_de_results_control_cells.h5ad"
    adata_filtered = ad.read_h5ad(filtered_data_path, backed=None)
    adata_filtered.obs['condition'] = adata_filtered.obs['condition'].astype(str)
    control_cells_ids_input = adata_filtered.uns["control_cells_ids"]

    cr4_data_path = (
        "/data2/czbenchmarks/replogle2022/raw_h5ad_from_cr4/K562_essential_mtx.h5ad"
    )
    adata_cr4 = ad.read_h5ad(cr4_data_path, backed=None)
    adata_cr4.var.index.name = "gene_id"
    adata_cr4.var.rename(columns={"gene_name": "gene"}, inplace=True)
    adata_cr4.obs.rename(columns={"gene_id": "condition"}, inplace=True)

    # num_conditions = 10
    # condition_list = np.asarray([x for x in control_cells_ids_input.keys() if x != 'non-targeting'])
    condition_list = np.asarray([x for x in adata_filtered.obs.condition.unique() if x != 'non-targeting'])
    # condition_list = np.random.choice(
    #     condition_list, size=num_conditions, replace=False
    # )
    # condition_list = list(condition_list)
    # new_control_cells_ids = {k: control_cells_ids_input[k] for k in condition_list}
    # control_cells_ids = new_control_cells_ids

    control_cells_ids_new = {}
    total_conditions = len(condition_list)
    with tqdm(
            total=total_conditions, desc="Processing conditions", unit="item"
        ) as pbar:
        for condition in condition_list:
            print(f"Getting matched controls for {condition}")
            pert_cells, matched_controls = get_matched_controls(
                adata=adata_cr4,
                perturbation=condition,
                min_cells=1,
                matchtype="GEM",
                verbose=False,
                dict_ctrls=None,
                dataset=None, # 'replogle'
                pert_column='condition',
                ctrl_condition='non-targeting',
                gem_column='gem_group',
                libsize_column='UMI_count',
            )
            if matched_controls:
                control_cells_ids_new[condition] = matched_controls
                print(f"Found {len(matched_controls)} matched controls for {condition}")
                
            pbar.set_postfix_str(f"Completed {pbar.n + 1}/{total_conditions}")
            pbar.update(1)

    print("Saving new control cells ids to json")
    with open("/data2/czbenchmarks/control_cells_ids_replogle_k562_essential_perturbpredict_validate.json", "w") as f:
        json.dump(control_cells_ids_new, f)



