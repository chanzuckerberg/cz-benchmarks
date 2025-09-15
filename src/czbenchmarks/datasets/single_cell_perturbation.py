import io
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm

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
        control_cells_ids (dict): Dictionary of control cell IDs matched to each condition.
        de_results (pd.DataFrame): Differential expression results calculated on ground truth data using matched controls.
        target_conditions_to_save (dict): Dictionary of target conditions for each cell.
    """

    control_matched_adata: ad.AnnData
    control_cells_ids: dict
    de_results: pd.DataFrame
    target_conditions_to_save: dict

    def __init__(
        self,
        path: Path,
        organism: Organism,
        condition_key: str = "condition",
        control_name: str = "ctrl",
        de_gene_col: str = "gene",
        deg_test_name: str = "wilcoxon",
        percent_genes_to_mask: float = 0.5,
        min_de_genes_to_mask: int = 5,
        pval_threshold: float = 1e-4,
        min_logfoldchange: float = 1.0,
        min_smd: float = 0.55,
        de_results_path: Optional[Path] = None,
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
                "ctrl".
            de_gene_col (str): Column name for the names of genes which are
                differentially expressed in the differential expression results.
                Defaults to "gene".
            deg_test_name (str): Name of the differential expression test
                condition. Options are "wilcoxon" or "t-test". Defaults to "wilcoxon".
            percent_genes_to_mask (float): Percentage of genes to mask.
                Default is 0.5.
            min_de_genes_to_mask (int): Minimum number of differentially
                expressed genes required to mask that condition. If not met, no genes
                are masked. Default is 5.
            pval_threshold (float): P-value threshold for differential expression.
                Default is 1e-4.
            min_logfoldchange (float): Minimum log-fold change for differential
                expression. Default is 1.0.
            min_smd (float): Minimum standardized mean difference for differential
                expression. Default is 0.55.
            de_results_path (Optional[Path]): Path to load differential expression
                results from csv file. If not provided, the deg data are used from
                adata.uns['de_results_{deg_test_name}'].
            task_inputs_dir (Optional[Path]): Directory for storing task-specific
                inputs.
        """
        super().__init__("single_cell_perturbation", path, organism, task_inputs_dir)
        self.condition_key = condition_key
        self.control_name = control_name
        self.deg_test_name = deg_test_name
        self.normalized_deg_test_name = deg_test_name.replace("-", "_")
        self.de_gene_col = de_gene_col
        self.percent_genes_to_mask = percent_genes_to_mask
        self.min_de_genes_to_mask = min_de_genes_to_mask
        self.pval_threshold = pval_threshold
        self.min_logfoldchange = min_logfoldchange
        self.min_smd = min_smd
        self.de_results_path = de_results_path

    def load_and_filter_deg_results(self):
        """
        Load and filter differential expression results.
        """
        if self.de_results_path:
            logger.info(f"Loading de_results from {self.de_results_path}")
            de_results = pd.read_csv(self.de_results_path)
        else:
            logger.info("Loading de_results from adata.uns")
            de_results = pd.DataFrame(
                self.adata.uns[f"de_results_{self.normalized_deg_test_name}"]
            )

        # Validate structure of deg data
        # TODO move column names to standardized location when utility function added
        filter_columns = ["pval_adj"]
        if self.deg_test_name == "wilcoxon":
            filter_columns.append("logfoldchange")
        else:
            filter_columns.append("standardized_mean_diff")

        error_str = ""
        warning_str = ""
        for col in filter_columns:
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

        if self.deg_test_name == "wilcoxon":
            filter_column = "logfoldchange"
            filter_criteria = self.min_logfoldchange

        elif self.deg_test_name == "t-test":
            filter_column = "standardized_mean_diff"
            filter_criteria = self.min_smd

        effect_mask = de_results[filter_column].abs() >= filter_criteria
        combined_mask = pval_mask & effect_mask
        filtered_rows_additional = (~combined_mask).sum() - filtered_rows_pval_threshold
        if filtered_rows_additional < 0:
            filtered_rows_additional = 0
        logger.info(
            f"Removed {filtered_rows_additional} rows of {len(de_results)} total rows using {filter_column} >= {filter_criteria}"
        )

        de_results = de_results[combined_mask]
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
            gene_col=self.de_gene_col,
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

        # Fast: condition -> integer row positions
        condition_to_indices = obs.groupby(self.condition_key, observed=True).indices

        # Fast: control ids -> integer row positions per condition (preserves order)
        control_to_indices = {
            cond: obs_index.get_indexer_for(ids)
            for cond, ids in self.control_cells_ids.items()
        }

        all_merged_data = []
        target_conditions_to_save = {}

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
                # target_conditions_to_save.update(result[1])
                for idx in adata_merged.obs.index:
                    target_conditions_to_save[idx] = target_condition_dict[
                        selected_condition
                    ]
                pbar.set_postfix_str(f"Completed {pbar.n + 1}/{total_conditions}")
                pbar.update(1)

        # Combine all adata objects
        logger.info(
            f"Collected {len(all_merged_data)} datasets for the sampled control-matched conditions."
        )
        adata_final = ad.concat(all_merged_data, index_unique=None)
        adata_final.obs[self.condition_key] = pd.Categorical(
            adata_final.obs[self.condition_key]
        )

        return adata_final, target_conditions_to_save

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

        if self.deg_test_name not in ["wilcoxon", "t-test"]:
            raise ValueError(
                f"Differential expression test name '{self.deg_test_name}' not supported. "
                "Options are 'wilcoxon' or 't-test'."
            )

        if self.de_results_path and not Path(self.de_results_path).exists():
            raise FileNotFoundError(
                f"Differential expression results path '{self.de_results_path}' not found"
            )
        else:
            if (
                f"de_results_{self.normalized_deg_test_name}"
                not in self.adata.uns.keys()
            ):
                raise ValueError(
                    f"Key 'de_results_{self.normalized_deg_test_name}' not found in adata.uns"
                )

        if "control_cells_ids" not in self.adata.uns.keys():
            raise ValueError("Key 'control_cells_ids' not found in adata.uns")

        # Load control_cells_ids from adata.uns
        self.control_cells_ids = self.adata.uns["control_cells_ids"]

        # Loading from h5ad file converts lists to numpy arrays
        for key in self.control_cells_ids.keys():
            self.control_cells_ids[key] = list(self.control_cells_ids[key])

        # Load and filter differential expression results
        logger.info(
            f"Loading and filtering differential expression results using {self.deg_test_name} test"
        )
        self.de_results = self.load_and_filter_deg_results()
        logger.info(f"Using {len(self.de_results)} differential expression values")

        # Compare conditions and throw warning or error for unmatched conditions
        unique_conditions_adata = set(self.adata.obs[self.condition_key])
        unique_conditions_control_cells_ids = set(self.control_cells_ids.keys())
        unique_conditions_de_results = set(self.de_results[self.condition_key])

        if not unique_conditions_de_results.issubset(unique_conditions_adata):
            raise ValueError(
                f"de_results[{self.condition_key}] contains conditions not in adata.obs[{self.condition_key}]. This will cause errors in the creation of the control-matched adata."
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
        adata_final, target_conditions_to_save = self._create_adata()

        self.control_matched_adata = adata_final
        self.target_conditions_to_save = target_conditions_to_save

    def store_task_inputs(self) -> Path:
        """
        Store auxiliary data files.

        This method saves the IDs of the control cells and the target conditions dictionary
            to JSON files.

        Returns:
            Path: Path to the directory storing the task input files.
        """
        # TODO: Might be better as a single adata, pending future design on how
        # Task instantiation is performed by benchmarking pipelines
        inputs_to_store = {
            "control_cells_ids": self.control_cells_ids,
            "target_conditions_to_save": self.target_conditions_to_save,
            "de_results": self.de_results,
            "control_matched_adata/obs": self.control_matched_adata.obs,
            "control_matched_adata/var": self.control_matched_adata.var,
            "control_matched_adata/X": self.control_matched_adata.X,
            "original_adata/obs/index": self.adata.obs.index.astype(str).to_numpy(),
        }

        for key, item in inputs_to_store.items():
            if hasattr(item, "to_json"):
                # For pandas DataFrames. Preserve index for obs/var by using orient="split".
                buffer = io.StringIO()
                if key in {"control_matched_adata/obs", "control_matched_adata/var"}:
                    item.to_json(buffer, orient="split")
                else:
                    item.to_json(buffer)
                self._store_task_input(f"{key}.json", buffer.getvalue())

            elif isinstance(item, np.ndarray):
                output_dir = self.task_inputs_dir / Path(key).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = self.task_inputs_dir / (key + ".npy")
                np.save(output_file, item)

            elif isinstance(item, sparse.csr_matrix):
                output_dir = self.task_inputs_dir / Path(key).parent
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = self.task_inputs_dir / (key + ".npz")
                sparse.save_npz(output_file, item)

            else:
                # For dictionaries and other JSON-serializable objects
                json_string = json.dumps(item)
                self._store_task_input(f"{key}.json", json_string)

        return self.task_inputs_dir

    def _validate(self) -> None:
        """
        Perform dataset-specific validation.

        Validates the following:
        - Condition format must be one of:
          - ``{control_name}`` or ``{control_name}_{perturb}`` for matched control samples.
          - ``{perturb}`` for single perturbations.
        - Combinatorial perturbations are not currently supported.

        Raises:
            ValueError: If invalid condition formats are found.
        """
        super()._validate()

        # Validate condition format
        conditions = set(self.control_matched_adata.obs[self.condition_key])
        target_conditions = set(
            x.split("_")[1] for x in self.target_conditions_to_save.keys()
        )  # Update for multiple perturbations

        for condition in conditions:
            if condition in target_conditions:
                continue
            elif condition.startswith(self.control_name):
                control_matched_condition = condition.split("_")[1]
                if control_matched_condition not in target_conditions:
                    raise ValueError(
                        f"Invalid control matched condition format: {condition}. "
                        f"Must be ``{self.control_name}`` or ``{self.control_name}_{{perturb}}``"
                    )
            else:
                # Update for multiple perturbations
                raise ValueError(
                    f"Invalid perturbation condition format: {condition}. "
                    f"Must be ``{self.control_name}`` or ``{self.control_name}_{{perturb}}`` for control samples,"
                    "or ``{perturb}`` for perturbations."
                )
