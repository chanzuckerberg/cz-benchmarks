from typing import Dict, Optional, List, Tuple
from typing import Dict, Optional, List, Tuple
import io
from pathlib import Path
import json
import numpy as np
import scipy.sparse as sparse
import pandas as pd
import anndata as ad
import logging
from czbenchmarks.datasets.single_cell import SingleCellDataset
from czbenchmarks.datasets.types import Organism
from ..constants import RANDOM_SEED

log = logging.getLogger(__name__)

def sample_de_genes(
    de_results: pd.DataFrame,
    percent_genes_to_mask: float,
    min_de_genes: int,
    condition_col: str,
    gene_col: str,
    seed: int = RANDOM_SEED,
) -> Dict[str, List[str]]:
    """
    Sample genes from a differential expression results dataframe.

    Args:
        de_results (pd.DataFrame): Differential expression results dataframe.
        percent_genes_to_mask (float): Percentage of genes to mask.
        min_de_genes (int): Minimum number of differentially expressed genes
            required to mask that condition. If not met, no genes are masked.
        condition_col (str): Column name for the condition.
        gene_col (str): Column name for the gene names.
        seed (int): Random seed.
    Returns:
        Dict[str, List[str]]: Dictionary of target genes and their sampled genes.
    """
    np.random.seed(seed)
    target_genes = de_results[condition_col].unique()
    target_gene_dict = {}
    for target in target_genes:
        gene_names = de_results[de_results[condition_col] == target][gene_col].values
        n_genes_to_sample = int(len(gene_names) * percent_genes_to_mask)
        if n_genes_to_sample >= min_de_genes:
            sampled_genes = np.random.choice(
                gene_names, size=n_genes_to_sample, replace=False
            ).tolist()
            target_gene_dict[target] = sampled_genes
    return target_gene_dict


class SingleCellPerturbationDataset(SingleCellDataset):
    """
    Single cell dataset with perturbation data, containing control and
    perturbed cells.

    This class extends `SingleCellDataset` to handle datasets with perturbation
    data. It includes functionality for validating condition formats,
    and storing perturbation truth data.

    Input data requirements:
    - H5AD file containing single cell gene expression data.
    - Must have a column ``{condition_name}`` in `adata.obs` specifying control and
      perturbed conditions.
    - Condition format must be one of:
      - ``{condition_name}`` for control samples.
      - ``{perturb}`` or ``{perturb}+{condition_name}`` for a single perturbation.
      - ``{perturb1}+{perturb2}`` for combinatorial perturbations.

    Attributes:
        control_cells_ids (dict): Dictionary of control cells IDs for each condition.
        target_genes_to_save (dict): Dictionary of target genes for each cell.
        de_results (pd.DataFrame): Differential expression results.
    """

    # FIXME MICHELLE add adata?
    control_cells_ids: dict
    target_genes_to_save: dict
    de_results: pd.DataFrame

    def __init__(
        self,
        path: Path,
        organism: Organism,
        condition_key: str = "condition",
        control_name: str = "ctrl",
        de_gene_col: str = "gene",
        deg_test_name: str = "wilcoxon",
        task_inputs_dir: Optional[Path] = None,
    ):
        """
        Initialize a SingleCellPerturbationDataset instance.

        Args:
            path (Path): Path to the dataset file.
            organism (Organism): Enum value indicating the organism.
            condition_key (str): Key for the column in `adata.obs` specifying conditions.
                Defaults to "condition".
            control_name (str): Name of the control condition. Defaults to "ctrl".
            de_gene_col (str): Column name for the gene names in the differential expression results.
                Defaults to "gene".
            deg_test_name (str): Name of the differential expression test condition.
                Options are "wilcoxon" or "t-test". Defaults to "wilcoxon".
            task_inputs_dir (Optional[Path]): Directory for storing task-specific inputs.
        """
        super().__init__("single_cell_perturbation", path, organism, task_inputs_dir)
        self.condition_key = condition_key
        self.control_name = control_name
        self.deg_test_name = deg_test_name
        self.de_gene_col = de_gene_col
        # FIXME MICHELLE add parameters for sampling genes?

    def _sample_genes_to_mask(
        self,
        percent_genes_to_mask: float = 0.5,
        min_de_genes: int = 5,
        pval_threshold: float = 1e-4,
        min_logfoldchange: float = 1.0,
        min_smd: float = 0.55,
    ) -> Dict[str, List[str]]:
        # FIXME MICHELLE double check default parameter values
        de_results = self.de_results.copy()
        
        filter = de_results["pval"] < pval_threshold

        if self.deg_test_name == "wilcoxon":
            filter &= de_results["logfoldchange"] > min_logfoldchange
            # filter &= de_results["num_de_genes"] >= min_de_genes # FIXME MICHELLE check if this should only be here
        elif self.deg_test_name == "t-test":
            filter &= de_results["standardized_mean_diff"] > min_smd

        de_results = de_results[filter]

        # TODO update DE results column names so they match the dataset column names
        target_gene_dict = sample_de_genes(
            de_results=de_results,
            percent_genes_to_mask=percent_genes_to_mask,
            min_de_genes=min_de_genes,
            condition_col=self.condition_key,
            gene_col=self.de_gene_col
        )

        return target_gene_dict

    def _create_adata(self) -> Tuple[ad.AnnData, dict]:
        target_gene_dict = self._sample_genes_to_mask()
        adata = self.adata
        control_cells_ids = self.control_cells_ids

        # Initialize list to store merged data
        all_merged_data = []

        # Initialize a dictionary to store target genes for each cell
        target_genes_to_save = {}

        target_genes = target_gene_dict.keys()
        # FIXME MICHELLE parallelize this loop
        for i, key in enumerate(target_genes):
            # Create condition and control data and update condition information
            adata_condition = adata[adata.obs[self.condition_key] == key].copy()
            adata_control = adata[adata.obs.index.isin(control_cells_ids[key])].copy()

            if len(adata_condition) != len(adata_control):
                log.warning(
                    f"Condition and control data for {key} have different lengths."
                )
                continue

            adata_condition.obs[self.condition_key] = adata_condition.obs[self.condition_key].astype(str)
            adata_condition.obs.loc[:, self.condition_key] = key
            
            adata_control.obs[self.condition_key] = adata_control.obs[self.condition_key].astype(str)
            adata_control.obs.loc[:, self.condition_key] = "_".join([self.control_name, key])

            # Merge condition and control data
            adata_merged = ad.concat([adata_condition, adata_control], index_unique=None)
            # adata_merged = adata_condition.concatenate(adata_control, index_unique=None) # FIXME MICHELLE REMOVE
            adata_merged.obs[self.condition_key] = pd.Categorical(adata_merged.obs[self.condition_key]) # FIXME MICHELLE MOVE TO AFTER MERGED?

            # Add new column cell_barcode_gene
            adata_merged.obs["cell_barcode_gene"] = (
                adata_merged.obs.index.astype(str) + "_" + [key] * len(adata_merged)
            )

            # Add target genes to the dictionary for each cell
            for idx in adata_merged.obs["cell_barcode_gene"]:
                target_genes_to_save[idx] = target_gene_dict[key]

            all_merged_data.append(adata_merged)

        # Combine all adata objects
        # adata_final = all_merged_data[0].concatenate(
        #     all_merged_data[1:], index_unique=None
        # )
        adata_final = ad.concat(all_merged_data, index_unique=None)

        # Set the new index
        adata_final.obs.set_index("cell_barcode_gene", inplace=True) # FIXME MICHELLE MOVE TO INSIDE LOOP?

        return adata_final, target_genes_to_save

    def load_data(self) -> None:
        """
        Load the dataset and populate perturbation truth data.

        This method validates the presence of `condition_key` in
        This method validates the presence of `condition_key` in
        `adata.obs`, and extracts control data for each condition into the
        `perturbation_truth` attribute.

        Raises:
            ValueError: If `condition_key` not found in `adata.obs`.
        """
        super().load_data()

        if self.condition_key not in self.adata.obs.columns:
            raise ValueError(
                f"Condition key '{self.condition_key}' not found in adata.obs"
            )

        if not self.adata.obs[self.condition_key].str.contains(self.control_name).any():
            raise ValueError(
                f"Data in condition key '{self.condition_key}' column does not contain control condition '{self.control_name}'"
            )

        if self.deg_test_name not in ["wilcoxon", "t-test"]:
            raise ValueError(
                f"Differential expression test name '{self.deg_test_name}' not supported. "
                "Options are 'wilcoxon' or 't-test'."
            )

        for key in ['control_cells_ids', f'de_results_{self.deg_test_name}']:
            if key not in self.adata.uns.keys():
                raise ValueError(f"Key '{key}' not found in adata.uns")

        self.control_cells_ids = self.adata.uns["control_cells_ids"]
        self.de_results = pd.DataFrame(
            self.adata.uns[f"de_results_{self.deg_test_name}"]
        )

        adata_final, target_genes_to_save = self._create_adata()

        self.adata = adata_final
        self.target_genes_to_save = target_genes_to_save

    def store_task_inputs(self) -> Path:
        """
        Store auxiliary data files.
        Store auxiliary data files.

        This method saves the IDs of the control cells and the target genes dictonary to JSON files.

        Returns:
            Path: Path to the directory storing the task input files.
        """
        # FIXME MICHELLE how to handle adata?
        inputs_to_store = {
            "control_cells_ids": self.control_cells_ids,
            "target_genes_to_save": self.target_genes_to_save,
            "de_results": self.de_results,
        }
        buffer = io.StringIO()
        
        for key, item in inputs_to_store.items():
            buffer = io.StringIO()
            item.to_json(buffer)
            self._store_task_input(f"{key}.json", buffer.getvalue())
        return self.task_inputs_dir

    def _validate(self) -> None:
        """
        Perform dataset-specific validation.

        Validates the following:
        - Condition format must be one of:
          - ``{condition_name}`` for control samples.
          - ``{perturb}`` or ``{perturb}+{condition_name}`` for single perturbations.
          - ``{perturb1}+{perturb2}`` for combinatorial perturbations.

        Raises:
            ValueError: If invalid condition formats are found.
            ValueError: If invalid condition formats are found.
        """
        super()._validate()

        # Validate condition format
        conditions = set(self.adata.obs[self.condition_key])

        for condition in conditions:
            if (
                condition in self.target_genes_to_save.keys()
            ) | condition == self.control_name:
                continue
            else:
                value_error = ValueError(
                    f"Invalid perturbation condition format: {condition}. "
                    f"Must be {self.control_name}, " + "'{perturb}', '{perturb}+"
                    f"{self.control_name}" + "', or '{perturb1}+{perturb2}'"
                )

                parts = condition.split("+")

                if len(parts) != 2:
                    raise value_error
                else:
                    for part in parts:
                        if (part not in self.target_genes_to_save.keys()) & (
                            part != self.control_name
                        ):
                            raise value_error
