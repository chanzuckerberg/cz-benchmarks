import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from czbenchmarks.constants import RANDOM_SEED
from czbenchmarks.datasets.single_cell import SingleCellDataset
from czbenchmarks.datasets.utils_single_cell import create_adata_for_condition
from czbenchmarks.datasets.types import Organism

logger = logging.getLogger(__name__)


def sample_de_genes(
    de_results: pd.DataFrame,
    percent_genes_to_mask: float,
    min_de_genes_to_mask: int,
    condition_col: str,
    gene_col: str,
    seed: int = RANDOM_SEED,
) -> Dict[str, List[str]]:
    """
    Sample genes from a differential expression results dataframe.

    Args:
        de_results (pd.DataFrame): Differential expression results dataframe.
        percent_genes_to_mask (float): Percentage of genes to mask.
        min_de_genes_to_mask (int): Minimum number of masked differentially
            expressed genes. If not met, no genes are masked.
        condition_col (str): Column name for the condition.
        gene_col (str): Column name for the gene names.
        seed (int): Random seed.
    Returns:
        Dict[str, List[str]]: Dictionary of target genes and their sampled genes.
    """
    np.random.seed(seed)
    target_conditions = de_results[condition_col].unique()
    target_condition_dict = {}
    for target in target_conditions:
        gene_names = de_results[de_results[condition_col] == target][gene_col].values
        n_genes_to_sample = int(len(gene_names) * percent_genes_to_mask)
        if n_genes_to_sample >= min_de_genes_to_mask:
            sampled_genes = np.random.choice(
                gene_names, size=n_genes_to_sample, replace=False
            ).tolist()
            target_condition_dict[target] = sampled_genes
    return target_condition_dict


class SingleCellPerturbationDataset(SingleCellDataset):
    """
    Single cell dataset with perturbation data, containing control and
    perturbed cells.

    This class extends `SingleCellDataset` to handle datasets with perturbation
    data. It includes functionality for validating condition formats,
    and perturbation data with matched control cells.

    Input data requirements:

    - H5AD file containing single-cell gene expression data.
    - Must have a column ``condition_key`` in ``adata.obs`` specifying control and perturbed conditions.
    - Condition format must be one of:

      - ``{control_name}`` or ``{control_name}_{perturb}`` for control samples.
      - ``{perturb}`` for a single perturbation.

    - Combinatorial (multiple) perturbations are not currently supported.

    Attributes:
        control_cells_ids (dict): Dictionary mapping each condition to a dictionary
            of treatment cell barcodes (keys) to matched control cell barcodes (values).
        de_results (pd.DataFrame): Differential expression results calculated on ground truth data using matched controls.
        target_conditions_dict (dict): Dictionary of masked genes for each condition.
    """

    control_matched_adata: ad.AnnData
    control_cells_ids: dict
    de_results: pd.DataFrame
    target_conditions_dict: dict

    def __init__(
        self,
        path: Path,
        organism: Organism,
        condition_key: str = "condition",
        control_name: str = "non-targeting",
        de_gene_col: str = "gene",
        percent_genes_to_mask: float = 0.5,
        min_de_genes_to_mask: int = 5,
        pval_threshold: float = 1e-4,
        min_logfoldchange: float = 1.0,
        task_inputs_dir: Optional[Path] = None,
    ):
        """
        Initialize a SingleCellPerturbationDataset instance.

        Args:
            path (Path): Path to the dataset file.
            organism (Organism): Enum value indicating the organism.
            condition_key (str): Key for the column in `adata.obs` specifying
                conditions. Defaults to "condition".
            control_name (str): Name of the control condition. Defaults to
                "non-targeting".
            de_gene_col (str): Column name for the names of genes which are
                differentially expressed in the differential expression results.
                Defaults to "gene".
            percent_genes_to_mask (float): Percentage of genes to mask.
                Default is 0.5.
            min_de_genes_to_mask (int): Minimum number of differentially
                expressed genes required to mask that condition. If not met, no genes
                are masked. Default is 5.
            pval_threshold (float): P-value threshold for differential expression.
                Default is 1e-4.
            min_logfoldchange (float): Minimum log-fold change for differential
                expression. Default is 1.0.
            task_inputs_dir (Optional[Path]): Directory for storing task-specific
                inputs.
        """
        super().__init__("single_cell_perturbation", path, organism, task_inputs_dir)
        self.condition_key = condition_key
        self.control_name = control_name
        self.deg_test_name = "wilcoxon"  # TODO: will allow other tests in the future
        self.de_gene_col = de_gene_col
        self.percent_genes_to_mask = percent_genes_to_mask
        self.min_de_genes_to_mask = min_de_genes_to_mask
        self.pval_threshold = pval_threshold
        self.min_logfoldchange = min_logfoldchange

    def load_and_filter_deg_results(self):
        """
        Load and filter differential expression results.
        """
        logger.info("Loading de_results from adata.uns")

        # FIXME MICHELLE: update to ensure proper handling of float precision
        de_results = pd.DataFrame(self.adata.uns[f"de_results_{self.deg_test_name}"])
        # de_results = pd.read_json(self.adata.uns[f"de_results_{self.deg_test_name}"], orient='records', precise_floats=True)

        # Validate structure of deg data
        error_str = ""
        warning_str = ""
        for col in ["pval_adj", "logfoldchange"]:
            if col not in de_results.columns:
                error_str += f"{col} column not found in de_results and required for {self.deg_test_name} test. "
            else:
                if de_results[col].isna().any():
                    warning_str += f"{col} column has missing or null values. "
        if len(warning_str) > 0:
            logger.warning(warning_str + "This may impact filtering of results.")
        if len(error_str) > 0:
            raise ValueError(error_str)

        # Perform filtering
        logger.info(f"Filtering de_results with pval_adj <= {self.pval_threshold}")
        pval_mask = de_results["pval_adj"] <= self.pval_threshold
        filtered_rows_pval_threshold = (~pval_mask).sum()
        logger.info(
            f"Removed {filtered_rows_pval_threshold} rows of {len(de_results)} total rows using pval_adj <= {self.pval_threshold}"
        )

        filter_column = "logfoldchange"
        effect_mask = de_results[filter_column].abs() >= self.min_logfoldchange
        combined_mask = pval_mask & effect_mask
        filtered_rows_additional = (~combined_mask).sum() - filtered_rows_pval_threshold
        if filtered_rows_additional < 0:
            filtered_rows_additional = 0
        logger.info(
            f"Removed {filtered_rows_additional} rows of {len(de_results)} total rows using {filter_column} >= {self.min_logfoldchange}"
        )

        de_results = de_results[combined_mask]
        if len(de_results) == 0:
            raise ValueError(
                "No differential expression results remain after filtering. "
                "Please check de data and filtering parameters."
            )
        return de_results

    def _create_adata(self) -> Tuple[ad.AnnData, dict]:
        """
        Create an AnnData object with perturbed and control cells.

        This method creates an AnnData object with perturbed and control cells,
        and adds target genes to the dictionary for each cell.
        """

        target_condition_dict = sample_de_genes(
            de_results=self.de_results,
            percent_genes_to_mask=self.percent_genes_to_mask,
            min_de_genes_to_mask=self.min_de_genes_to_mask,
            condition_col=self.condition_key,
            gene_col="gene_id",  # Column was renamed to gene_id during optimization
        )

        target_conditions = list(target_condition_dict.keys())
        total_conditions = len(target_conditions)
        logger.info(f"Sampled {total_conditions} conditions for masking")

        # Do this once before the loop
        obs = self.adata.obs
        obs_index = obs.index

        # If not already categorical, this speeds grouping and comparisons
        if not isinstance(obs[self.condition_key], pd.CategoricalDtype):
            obs[self.condition_key] = pd.Categorical(obs[self.condition_key])

        # FIXME MICHELLE validate existence of ids from control_cells_ids in adata.obs

        # Experimental ids -> integer row positions per condition and preserves order
        # FIXME MICHELLE can use experimental ids from target_condition_dict bc these
        # are the only cells that data is needed for?
        condition_to_indices = {
            cond: obs_index.get_indexer_for(list(mapping.keys()))
            for cond, mapping in self.control_cells_ids.items()
        }

        # Control ids -> integer row positions per condition and preserves order
        control_to_indices = {
            cond: obs_index.get_indexer_for(list(mapping.values()))
            for cond, mapping in self.control_cells_ids.items()
        }

        all_merged_data = []

        with tqdm(
            total=total_conditions, desc="Processing conditions", unit="item"
        ) as pbar:
            for selected_condition in target_conditions:
                adata_merged, _ = create_adata_for_condition(
                    adata=self.adata,
                    condition=selected_condition,
                    condition_key=self.condition_key,
                    control_name=self.control_name,
                    rows_cond=condition_to_indices[selected_condition],
                    rows_ctrl=control_to_indices[selected_condition],
                )

                all_merged_data.append(adata_merged)
                pbar.set_postfix_str(f"Completed {pbar.n + 1}/{total_conditions}")
                pbar.update(1)

        # Combine all adata objects
        logger.info(
            f"Merged datasets for {len(all_merged_data)} control-matched conditions."
        )
        adata_final = ad.concat(all_merged_data, index_unique=None)
        adata_final.obs[self.condition_key] = pd.Categorical(
            adata_final.obs[self.condition_key]
        )

        # Optimize: Keep only necessary columns in obs (only condition_key is used in task)
        adata_final.obs = adata_final.obs[[self.condition_key]]

        # Add task-related data to uns for easy access
        adata_final.uns["target_conditions_dict"] = target_condition_dict
        adata_final.uns["de_results"] = {
            col: self.de_results[col].values for col in self.de_results.columns
        }
        adata_final.uns["cell_barcode_condition_index"] = self.adata.obs.index.astype(
            str
        ).values
        adata_final.uns["control_cells_ids"] = self.control_cells_ids

        return adata_final, target_condition_dict

    def load_data(
        self,
    ) -> None:
        """
        Load the dataset and populate perturbation truth data.

        This method validates the presence of `condition_key` in
        `adata.obs`, and extracts control data for each condition into the
        `perturbation_truth` attribute.

        Raises:
            ValueErrors or FileNotFoundErrors based on required data structure.
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

        if f"de_results_{self.deg_test_name}" not in self.adata.uns.keys():
            raise ValueError(
                f"Key 'de_results_{self.deg_test_name}' not found in adata.uns"
            )

        if "control_cells_ids" not in self.adata.uns.keys():
            raise ValueError("Key 'control_cells_ids' not found in adata.uns")

        # Load control_cells_ids from adata.uns
        self.control_cells_ids = self.adata.uns["control_cells_ids"]

        # Load and filter differential expression results
        logger.info(
            f"Loading and filtering differential expression results using {self.deg_test_name} test"
        )
        self.de_results = self.load_and_filter_deg_results()
        logger.info(f"Using {len(self.de_results)} differential expression values")

        # Optimize: Keep only necessary columns in de_results
        # Task only uses: condition_key, "gene_id", and metric_column (logfoldchange or standardized_mean_diff)
        metric_column = (
            "logfoldchange"
            if self.deg_test_name == "wilcoxon"
            else "standardized_mean_diff"
        )
        necessary_columns = [self.condition_key, self.de_gene_col, metric_column]

        # Ensure we have gene_id column for compatibility with task
        if self.de_gene_col != "gene_id":
            self.de_results = self.de_results.rename(
                columns={self.de_gene_col: "gene_id"}
            )
            necessary_columns = [self.condition_key, "gene_id", metric_column]

        self.de_results = self.de_results[necessary_columns]

        # Compare conditions and throw warning or error for unmatched conditions
        unique_conditions_adata = set(self.adata.obs[self.condition_key])
        unique_conditions_control_cells_ids = set(self.control_cells_ids.keys())
        unique_conditions_de_results = set(self.de_results[self.condition_key])

        if self.control_name in unique_conditions_adata:
            unique_conditions_adata.remove(self.control_name)

        if not unique_conditions_de_results.issubset(unique_conditions_adata):
            raise ValueError(
                f"de_results[{self.condition_key}] contains conditions not in adata.obs[{self.condition_key}]. "
                "This will cause errors in the creation of the control-matched adata."
            )

        if not unique_conditions_de_results.issubset(
            unique_conditions_control_cells_ids
        ):
            raise ValueError(
                f"Conditions in de_results[{self.condition_key}] are not a subset "
                f"of control_cells_ids keys. This will cause errors in the "
                f"creation of the control-matched adata."
            )

        if unique_conditions_control_cells_ids != unique_conditions_adata:
            msg = (
                f"Conditions in control_cells_ids and adata.obs[{self.condition_key}] "
                f"are not identical"
            )
            if unique_conditions_control_cells_ids.issubset(unique_conditions_adata):
                logger.warning(
                    msg + f", but control_cells_ids keys are a subset of "
                    f"adata.obs[{self.condition_key}]. This should allow for "
                    f"creation of control-matched data but will ignore some of "
                    f"the data"
                )
            else:
                logger.warning(
                    msg + f", and control_cells_ids keys contain conditions not in "
                    f"adata.obs[{self.condition_key}]. This may cause errors in "
                    f"the creation of control-matched adata."
                )

        logger.info(
            f"Creating control-matched adata for {len(self.control_cells_ids)} conditions"
        )
        adata_final, target_conditions_dict = self._create_adata()

        self.control_matched_adata = adata_final
        self.target_conditions_dict = target_conditions_dict

    def store_task_inputs(self) -> Path:
        """
        Store all task inputs as separate files.

        This method saves all task-related data as separate files:
        - control_matched_adata.h5ad: The main AnnData object (includes cell_barcode_condition_index, control_cells_ids, target_conditions_dict, and de_results in uns)
        - target_conditions_dict.json: Target conditions dictionary
        - de_results.parquet: Differential expression results

        Returns:
            Path: Path to the task inputs directory.
        """
        # Ensure the task inputs directory exists
        self.task_inputs_dir.mkdir(parents=True, exist_ok=True)
        adata_to_save = self.control_matched_adata.copy()
        adata_to_save.uns["cell_barcode_condition_index"] = self.adata.obs.index.astype(
            str
        ).values

        # Save the main AnnData object
        adata_file = self.task_inputs_dir / "control_matched_adata.h5ad"
        adata_to_save.write_h5ad(adata_file)

        # Save target conditions dict as JSON
        target_conditions_file = self.task_inputs_dir / "target_conditions_dict.json"
        with open(target_conditions_file, "w") as f:
            json.dump(self.target_conditions_dict, f)

        # Save DE results as Parquet using PyArrow
        de_results_file = self.task_inputs_dir / "de_results.parquet"
        self.de_results.to_parquet(de_results_file, engine="pyarrow", index=False)

        return self.task_inputs_dir

    def _validate(self) -> None:
        """
        Perform dataset-specific validation.

        Validates the following:
        - Condition format must be one of:
          - ``{control_name}`` or ``{control_name}_{perturb}`` for unmatched or matched control samples.
          - ``{perturb}`` for single perturbations.
        - Combinatorial perturbations are not currently supported.

        Raises:
            ValueError: If invalid condition formats are found.
        """
        super()._validate()

        if self.condition_key not in self.adata.obs.columns:
            raise ValueError(
                f"Condition key '{self.condition_key}' not found in adata.obs"
            )
        if self.condition_key not in self.control_matched_adata.obs.columns:
            raise ValueError(
                f"Condition key '{self.condition_key}' not found in control_matched_adata.obs"
            )

        # Validate matched_adata condition format by checking the ORIGINAL conditions before processing
        original_conditions = set(self.control_matched_adata.obs[self.condition_key])
        target_conditions = set(self.target_conditions_dict.keys())
        for condition in original_conditions:
            # Check if it's a valid perturbation condition (just the perturbation name)
            if condition in target_conditions:
                continue
            # Check if it's a control condition: just control_name
            elif condition == self.control_name:
                continue
            # Check if it's a matched control condition: control_name_perturb
            elif condition.startswith(f"{self.control_name}_"):
                # Extract the perturbation part after control_name_
                perturb_part = condition[len(f"{self.control_name}_") :]
                if perturb_part in target_conditions:
                    continue
                else:
                    raise ValueError(
                        f"Invalid matched control condition format: {condition}. "
                        f"The perturbation part '{perturb_part}' is not a valid target condition. "
                        f"Valid target conditions: {list(target_conditions)}"
                    )
            else:
                # Invalid condition format
                raise ValueError(
                    f"Invalid condition format: {condition}. "
                    f"Must be one of:\n"
                    f"  - ``{self.control_name}`` for unmatched control samples\n"
                    f"  - ``{self.control_name}_{{perturb}}`` for matched control samples\n"
                    f"  - ``{{perturb}}`` for single perturbations (where perturb is one of {list(target_conditions)})"
                )
