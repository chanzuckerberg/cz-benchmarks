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
from czbenchmarks.constants import RANDOM_SEED
from tqdm import tqdm
from time import time
from anndata import ImplicitModificationWarning
import warnings

# warnings.filterwarnings("ignore", category=ImplicitModificationWarning) # FIXME MICHELLE: for testing, remove

logger = logging.getLogger(__name__)


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
        target_genes_to_save (dict): Dictionary of target genes for each cell.
    """

    control_matched_adata: ad.AnnData
    control_cells_ids: dict
    de_results: pd.DataFrame
    target_genes_to_save: dict
    # FIXME MICHELLE verify variable naming for future inclusion of chemical perturbation datasets

    def __init__(
        self,
        path: Path,
        organism: Organism,
        condition_key: str = "condition",
        control_name: str = "ctrl",
        de_gene_col: str = "gene",
        deg_test_name: str = "wilcoxon",
        percent_genes_to_mask: float = 1.0,
        min_de_genes: int = 1, # FIXME MICHELLE: Maria had 5,Jasleen suggested 1
        pval_threshold: float = 1e-2, # FIXME MICHELLE: Maria had 1e-4, Jasleen suggested 1e-2 or 5e-2
        min_logfoldchange: float = 1.0, # FIXME MICHELLE: should this be changed to >= 0?
        min_smd: float = 0.55,
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
            de_gene_col (str): Column name for the names of genes which are differentially
                expressed in the differential expression results. Defaults to "gene".
            deg_test_name (str): Name of the differential expression test condition.
                Options are "wilcoxon" or "t-test". Defaults to "wilcoxon".
            percent_genes_to_mask (float): Percentage of genes to mask. Defaults to 0.5.
            min_de_genes (int): Minimum number of differentially expressed genes
                required to mask that condition. If not met, no genes are masked.
            pval_threshold (float): P-value threshold for differential expression.
            min_logfoldchange (float): Minimum log-fold change for differential expression.
            min_smd (float): Minimum standardized mean difference for differential expression.
            task_inputs_dir (Optional[Path]): Directory for storing task-specific inputs.
        """
        super().__init__("single_cell_perturbation", path, organism, task_inputs_dir)
        self.condition_key = condition_key
        self.control_name = control_name
        self.deg_test_name = deg_test_name
        self.de_gene_col = de_gene_col
        self.percent_genes_to_mask = percent_genes_to_mask
        self.min_de_genes = min_de_genes
        self.pval_threshold = pval_threshold
        self.min_logfoldchange = min_logfoldchange
        self.min_smd = min_smd

    def _create_adata(self) -> Tuple[ad.AnnData, dict]:
        """
        Create an AnnData object with perturbed and control cells.

        This method creates an AnnData object with perturbed and control cells,
        and adds target genes to the dictionary for each cell.
        """

        def _create_adata_for_condition(
            selected_condition: str,
            target_gene_dict: dict,
            rows_cond: np.ndarray,
            rows_ctrl: np.ndarray,
            OPTIMIZED: bool,
            adata: ad.AnnData = self.adata,
            control_cells_ids: dict = self.control_cells_ids,
            condition_key: str = "condition",
            control_name: str = "ctrl",
        ):
            """
            Create an AnnData object for a single condition.
            Setup as a private function to allow for multiprocessing if needed.
            """

            start_time = time()
            if not OPTIMIZED:
                adata_condition = adata[
                    adata.obs[self.condition_key] == selected_condition
                ].copy()
                adata_control = adata[
                    adata.obs.index.isin(control_cells_ids[selected_condition])
                ].copy()
            else:
                adata_condition = adata[rows_cond]
                adata_control = adata[rows_ctrl]
            index_time = time() - start_time

            if len(adata_condition) != len(adata_control):
                logger.warning(
                    f"Condition and control data for {selected_condition} have different lengths."
                )

            if not OPTIMIZED:
                adata_condition.obs[condition_key] = adata_condition.obs[
                    condition_key
                ].astype(str)
                adata_condition.obs.loc[:, condition_key] = selected_condition

                adata_control.obs[condition_key] = adata_control.obs[condition_key].astype(
                    str
                )
                adata_control.obs.loc[:, condition_key] = "_".join(
                    [control_name, selected_condition]
                )
            else:
                pass
                # label_cond = selected_condition
                # label_ctrl = f"{control_name}_{selected_condition}"

                # for a, label in ((adata_condition, label_cond), (adata_control, label_ctrl)):
                #     # if pd.api.types.is_categorical_dtype(a.obs[condition_key]):
                #     #     if label not in a.obs[condition_key].cat.categories:
                #     #         a.obs[condition_key] = a.obs[condition_key].cat.add_categories([label])
                #     a.obs[condition_key] = label

            # Concatenate condition and control data
            start_time = time()
            if not OPTIMIZED:
                adata_merged = ad.concat(
                    [adata_condition, adata_control], index_unique=None
                ).copy()
            else:
                adata_merged = ad.concat(
                    [adata_condition, adata_control], index_unique=None
                ).copy()
            small_concat_time = time() - start_time

            if OPTIMIZED:
                label_cond = [selected_condition] * len(adata_condition)
                label_ctrl = [f"{control_name}_{selected_condition}"] * len(adata_control)
                adata_merged.obs[condition_key] = label_cond + label_ctrl

            # Add condition to cell_barcode_gene column and set as index
            if not OPTIMIZED:
                adata_merged.obs["cell_barcode_gene"] = (
                    adata_merged.obs.index.astype(str)
                    + "_"
                    + [selected_condition] * len(adata_merged)
                )
                adata_merged.obs.set_index("cell_barcode_gene", inplace=True)
            else:
                adata_merged.obs_names = adata_merged.obs_names.astype(str) + "_" + selected_condition

            # Add target genes to the dictionary for each cell
            target_genes_to_save = {}
            for idx in adata_merged.obs.index:
                target_genes_to_save[idx] = target_gene_dict[selected_condition]

            return adata_merged, target_genes_to_save, small_concat_time, index_time

        target_gene_dict = sample_de_genes(
            de_results=self.de_results,
            percent_genes_to_mask=self.percent_genes_to_mask,
            min_de_genes=self.min_de_genes,
            condition_col=self.condition_key,
            gene_col=self.de_gene_col,
        )

        target_genes = list(target_gene_dict.keys())
        if self.num_conditions is not None: # FIXME MICHELLE: for testing, remove
            if self.num_conditions < len(target_genes):
                target_genes = target_genes[:self.num_conditions]
            else:
                logger.warning(f"num_conditions {self.num_conditions} is greater than the number of conditions in the dataset {len(target_genes)}. Using all conditions.")
        total_conditions = len(target_genes)
        logger.info(f"Sampled {total_conditions} conditions for masking")

        # Do this once before the loop
        obs = self.adata.obs
        obs_index = obs.index

        # If not already categorical, this speeds grouping and comparisons
        if not pd.api.types.is_categorical_dtype(obs[self.condition_key]):
            obs[self.condition_key] = pd.Categorical(obs[self.condition_key])

        # Fast: condition -> integer row positions
        condition_to_indices = obs.groupby(self.condition_key, observed=True).indices

        # Fast: control ids -> integer row positions per condition (preserves order)
        control_to_indices = {
            cond: obs_index.get_indexer_for(ids)
            for cond, ids in self.control_cells_ids.items()
        }

        all_merged_data = []
        target_genes_to_save = {}
        small_concat_time = 0.0
        index_time = 0.0

        with tqdm(
            total=total_conditions,
            desc="Processing conditions",
            unit="item"
        ) as pbar:
            for selected_condition in target_genes:
                result = _create_adata_for_condition(selected_condition=selected_condition, 
                                                    target_gene_dict=target_gene_dict, 
                                                    rows_cond=condition_to_indices[selected_condition], 
                                                    rows_ctrl=control_to_indices[selected_condition],
                                                    OPTIMIZED=self.OPTIMIZED)
                all_merged_data.append(result[0])
                target_genes_to_save.update(result[1])
                small_concat_time += result[2]
                index_time += result[3]
                pbar.set_postfix_str(f"Completed {pbar.n + 1}/{total_conditions}")
                pbar.update(1)

        # Combine all adata objects
        logger.info(f"Index time: {index_time:.2f} seconds")
        logger.info(f"Concat time: {small_concat_time:.2f} seconds")
        logger.info(f"Collected {len(all_merged_data)} datasets for the sampled control-matched conditions.")
        start_time = time()
        adata_final = ad.concat(all_merged_data, index_unique=None)
        logger.info(f"Large concat time: {time() - start_time:.2f} seconds")
        adata_final.obs[self.condition_key] = pd.Categorical(
            adata_final.obs[self.condition_key]
        )

        return adata_final, target_genes_to_save

    def load_data(self, OPTIMIZED: bool, num_conditions: Optional[int] = None) -> None: # FIXME MICHELLE: for testing, remove
        """
        Load the dataset and populate perturbation truth data.

        This method validates the presence of `condition_key` in
        `adata.obs`, and extracts control data for each condition into the
        `perturbation_truth` attribute.

        Raises:
            ValueError: If `condition_key` not found in `adata.obs`.
        """
        logger.info(f"Loading dataset with OPTIMIZED={OPTIMIZED} and num_conditions={num_conditions}")
        self.OPTIMIZED = OPTIMIZED
        self.num_conditions = num_conditions
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

        for key in ["control_cells_ids", f"de_results_{self.deg_test_name}"]:
            if key not in self.adata.uns.keys():
                raise ValueError(f"Key '{key}' not found in adata.uns")

        self.control_cells_ids = self.adata.uns["control_cells_ids"]

        # Loading from h5ad file converts lists to numpy arrays
        for key in self.control_cells_ids.keys():
            self.control_cells_ids[key] = list(self.control_cells_ids[key])

        # FIXME MICHELLE: add option to load de_results from file, 
        # add logging for filtering of de_results, move to separate method
        de_results = pd.DataFrame(
            self.adata.uns[f"de_results_{self.deg_test_name}"]
        )

        # FIXME check column names for de_results

        filter = de_results["pval_adj"] <= self.pval_threshold

        if self.deg_test_name == "wilcoxon":
            filter &= (
                de_results["logfoldchange"].abs() >= self.min_logfoldchange
            )
        elif self.deg_test_name == "t-test":
            filter &= de_results["standardized_mean_diff"].abs() >= self.min_smd

        self.de_results = de_results[filter]

        logger.info(f"Creating adata for {len(self.control_cells_ids)} conditions")
        start_time = time()
        adata_final, target_genes_to_save = self._create_adata()
        logger.info(f"Time to create control-matched adata: {time() - start_time:.2f} seconds")

        self.control_matched_adata = adata_final
        self.target_genes_to_save = target_genes_to_save

    def store_task_inputs(self) -> Path:
        """
        Store auxiliary data files.

        This method saves the IDs of the control cells and the target genes dictonary to JSON files.

        Returns:
            Path: Path to the directory storing the task input files.
        """
        inputs_to_store = {
            "control_cells_ids": self.control_cells_ids,
            "target_genes_to_save": self.target_genes_to_save,
            "de_results": self.de_results,
            "control_matched_adata/obs": self.control_matched_adata.obs,
            "control_matched_adata/var": self.control_matched_adata.var,
            "control_matched_adata/X": self.control_matched_adata.X,
        }

        for key, item in inputs_to_store.items():
            if hasattr(item, "to_json"):
                # For pandas DataFrames
                buffer = io.StringIO()
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
            x.split("_")[1] for x in self.target_genes_to_save.keys()
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
