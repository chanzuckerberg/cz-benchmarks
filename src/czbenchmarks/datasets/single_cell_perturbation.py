import logging
from pathlib import Path
from typing import Dict, List, Optional

import anndata as ad
import numpy as np
import pandas as pd
from czbenchmarks.constants import RANDOM_SEED
from czbenchmarks.datasets.single_cell import SingleCellDataset
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
    Sample a percentage of genes for masking for each condition from a
    differential expression results dataframe.

    Args:
        de_results (pd.DataFrame): Differential expression results dataframe.
        percent_genes_to_mask (float): Percentage of genes to mask.
        min_de_genes_to_mask (int): Minimum number of masked differentially
            expressed genes. If not met, no genes are masked.
        condition_col (str): Column name for the condition.
        gene_col (str): Column name for the gene names.
        seed (int): Random seed.
    Returns:
        Dict[str, List[str]]: Dictionary that maps each condition to a list of
        genes to be masked for that condition.
    """
    # Validate parameters
    for param_name, param in zip(
        [
            "percent_genes_to_mask",
            "min_de_genes_to_mask",
        ],
        [
            percent_genes_to_mask,
            min_de_genes_to_mask,
        ],
    ):
        if param < 0.0:
            raise ValueError(
                f"Parameter {param_name} must be greater than 0.0, got {param}"
            )

    np.random.seed(seed)
    target_conditions = de_results[condition_col].unique()
    target_conditions_dict = {}
    for target in target_conditions:
        gene_names = de_results[de_results[condition_col] == target][gene_col].values
        n_genes_to_sample = int(len(gene_names) * percent_genes_to_mask)
        if n_genes_to_sample >= min_de_genes_to_mask:
            sampled_genes = np.random.choice(
                gene_names, size=n_genes_to_sample, replace=False
            ).tolist()
            target_conditions_dict[target] = sampled_genes
    return target_conditions_dict


class SingleCellPerturbationDataset(SingleCellDataset):
    """
    Single cell dataset with perturbation data, containing control and
    perturbed cells.

    This class extends `SingleCellDataset` to handle datasets with perturbation
    data. It includes functionality for validating condition formats,
    and perturbation data with matched control cells.

    Input data requirements:

    - H5AD file containing single-cell gene expression data.
    - Must have a column ``condition_key`` in ``adata.obs`` specifying
        control and perturbed conditions.
    - Condition format must be one of:
      - ``{control_name}`` for control samples.
      - ``{perturb}`` for a single perturbation.

    Attributes:
        de_results (pd.DataFrame): Differential expression results calculated on ground
            truth data using matched controls.
        target_conditions_dict (Dict[str, List[str]]): Dictionary that maps each
            condition to a list of masked genes for that condition.
        control_cells_ids (dict): Dictionary mapping each condition to a dictionary
            of treatment cell barcodes (keys) to matched control cell barcodes (values).
            It is used primarily for creation of differential expression results
            in data processing and may be removed in a future release.
    """

    de_results: pd.DataFrame
    target_conditions_dict: dict
    control_cells_ids: dict

    # UNS keys contract for task consumption
    UNS_DE_RESULTS_KEY = "de_results"
    UNS_CONTROL_MAP_KEY = "control_cells_map"
    UNS_TARGET_GENES_KEY = "target_conditions_dict"
    UNS_METRIC_COL_KEY = "metric_column"
    UNS_CONFIG_KEY = "config"
    UNS_RANDOM_SEED_KEY = "random_seed"

    def __init__(
        self,
        path: Path,
        organism: Organism,
        condition_key: str = "condition",
        control_name: str = "ctrl",
        de_gene_col: str = "gene",
        de_metric_col: str = "logfoldchange",
        de_pval_col: str = "pval_adj",
        percent_genes_to_mask: float = 0.5,
        min_de_genes_to_mask: int = 5,
        pval_threshold: float = 1e-4,
        min_logfoldchange: float = 1.0,
        task_inputs_dir: Optional[Path] = None,
        random_seed: int = RANDOM_SEED,
        target_conditions_override: Optional[Dict[str, List[str]]] = None,
    ):
        """
        Instantiate a SingleCellPerturbationDataset instance.

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
            de_metric_col (str): Column name for the metric of the differential expression results.
                Defaults to "logfoldchange".
            de_pval_col (str): Column name for the p-value of the differential expression results.
                Defaults to "pval_adj".
            percent_genes_to_mask (float): Percentage of genes to mask.
                Default is 0.5.
            min_de_genes_to_mask (int): Minimum number of differentially
                expressed genes required to mask that condition. If not met, no genes
                are masked. Default is 5.
            pval_threshold (float): P-value threshold for differential expression.
                Default is 1e-4.
            min_logfoldchange (float): Minimum log-fold change for differential
                expression. Default is 1.0.
            task_inputs_dir (Optional[Path]): Path to the directory containing the task inputs.
                Default is None. If not provided, a default path will be used.
            random_seed (int): Random seed for reproducibility.
            target_conditions_override (Optional[Dict[str, List[str]]]): Dictionary that
                maps a target condition to a list of genes that the user specified to be masked.
                This overrides the default sampling of genes for masking in target_conditions_dict.
                Default is None.
        """
        super().__init__("single_cell_perturbation", path, organism, task_inputs_dir)

        if random_seed < 0:
            raise ValueError(
                f"Parameter random_seed must be greater than 0, got {random_seed}"
            )
        self.random_seed = random_seed
        self.condition_key = condition_key
        self.control_name = control_name
        self.deg_test_name = "wilcoxon"  # TODO: will allow other tests in the future
        self.de_gene_col = de_gene_col
        self.de_metric_col = de_metric_col
        self.de_pval_col = de_pval_col
        self.target_conditions_override = target_conditions_override
        self.percent_genes_to_mask = percent_genes_to_mask
        self.min_de_genes_to_mask = min_de_genes_to_mask
        self.pval_threshold = pval_threshold
        self.min_logfoldchange = min_logfoldchange

    def load_and_filter_deg_results(self):
        """
        Load and filter differential expression results from adata.uns.
        - Enforces that de_pval_col and de_metric_col are present in the dataframe and are not null.
        - Filters out rows where the p-value is greater than the pval_threshold.
        - Filters out rows where the metric is less than the min_logfoldchange.
        - Returns the filtered dataframe.

        Returns:
            pd.DataFrame: Differential expression results dataframe after filtering.
        """
        logger.info("Loading de_results from adata.uns")
        de_results = pd.DataFrame(self.adata.uns[f"de_results_{self.deg_test_name}"])

        # Validate structure of deg data
        error_str = ""
        warning_str = ""
        for col in [self.de_pval_col, self.de_metric_col]:
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
        logger.info(
            f"Filtering de_results with {self.de_pval_col} <= {self.pval_threshold}"
        )
        pval_mask = de_results[self.de_pval_col] < self.pval_threshold
        filtered_rows_pval_threshold = (~pval_mask).sum()
        logger.info(
            f"Removed {filtered_rows_pval_threshold} rows of {len(de_results)} total rows using pval_adj <= {self.pval_threshold}"
        )

        filter_column = self.de_metric_col
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

    def _populate_task_uns(self, de_results_df: pd.DataFrame) -> None:
        """Populate self.adata.uns with the inputs required by the perturbation task.

        - Stores filtered differential expression results under uns['de_results']
        - Optionally stores control mapping under uns['control_cells_map'] if present
        - Stores (or computes) target_conditions_dict under uns['target_conditions_dict']
        - Writes uns['metric_column'] with the configured metric name
        """
        metric_column = self.de_metric_col
        necessary_columns = [self.condition_key, "gene_id", metric_column]
        de_results_min = de_results_df[necessary_columns].copy()

        # Persist minimal de results in uns as a DataFrame
        self.adata.uns[self.UNS_DE_RESULTS_KEY] = de_results_min
        # Persist metric column name for downstream consumers
        self.adata.uns[self.UNS_METRIC_COL_KEY] = metric_column

        # Control mapping is REQUIRED and must be strict 1-1: condition -> dict(treated_barcode -> control_barcode)
        existing_map = self.adata.uns.get(self.UNS_CONTROL_MAP_KEY, None)
        if not (isinstance(existing_map, dict) and len(existing_map) > 0):
            raise ValueError(
                f"Missing required strict control mapping in adata.uns['{self.UNS_CONTROL_MAP_KEY}']"
            )
        # Local reference
        self.control_cells_ids = self.adata.uns[self.UNS_CONTROL_MAP_KEY]

        # Determine target_conditions_dict: override, or sample deterministically
        if self.target_conditions_override is not None:
            target_conditions_dict = self.target_conditions_override
        else:
            target_conditions_dict = sample_de_genes(
                de_results=de_results_min,
                percent_genes_to_mask=self.percent_genes_to_mask,
                min_de_genes_to_mask=self.min_de_genes_to_mask,
                condition_col=self.condition_key,
                gene_col="gene_id",
                seed=self.random_seed,
            )
        self.target_conditions_dict = target_conditions_dict
        self.adata.uns[self.UNS_TARGET_GENES_KEY] = target_conditions_dict

        # Persist config and seed for provenance
        self.adata.uns[self.UNS_RANDOM_SEED_KEY] = int(self.random_seed)
        self.adata.uns[self.UNS_CONFIG_KEY] = {
            "condition_key": self.condition_key,
            "control_name": self.control_name,
            "de_gene_col": self.de_gene_col,
            "de_metric_col": self.de_metric_col,
            "de_pval_col": self.de_pval_col,
            "pval_threshold": float(self.pval_threshold),
            "min_logfoldchange": float(self.min_logfoldchange),
            "percent_genes_to_mask": float(self.percent_genes_to_mask),
            "min_de_genes_to_mask": int(self.min_de_genes_to_mask),
        }

    def _validate_uns_for_task(self, adata: ad.AnnData) -> None:
        """Validate presence and format of required keys in adata.uns for the task."""
        if self.UNS_DE_RESULTS_KEY not in adata.uns:
            raise ValueError(
                f"Missing adata.uns['{self.UNS_DE_RESULTS_KEY}'] for perturbation task"
            )
        de_results = adata.uns[self.UNS_DE_RESULTS_KEY]
        if not isinstance(de_results, pd.DataFrame):
            raise ValueError(
                f"adata.uns['{self.UNS_DE_RESULTS_KEY}'] must be a pandas DataFrame"
            )
        metric_col = self.adata.uns.get(self.UNS_METRIC_COL_KEY, self.de_metric_col)
        for col in [self.condition_key, "gene_id", metric_col]:
            if col not in de_results.columns:
                raise ValueError(
                    f"adata.uns['{self.UNS_DE_RESULTS_KEY}'] missing required column '{col}'"
                )

        # Control mapping is REQUIRED at dataset level and must be strict 1-1
        if self.UNS_CONTROL_MAP_KEY not in adata.uns:
            raise ValueError(
                f"Missing adata.uns['{self.UNS_CONTROL_MAP_KEY}'] for perturbation task"
            )
        cm = adata.uns[self.UNS_CONTROL_MAP_KEY]
        if not isinstance(cm, dict):
            raise ValueError(f"adata.uns['{self.UNS_CONTROL_MAP_KEY}'] must be a dict")
        for cond, mapping in cm.items():
            if not isinstance(mapping, dict):
                raise ValueError(
                    f"{self.UNS_CONTROL_MAP_KEY}['{cond}'] must be a dict of treated_barcode -> control_barcode"
                )
            for tb, ctl in mapping.items():
                if not isinstance(tb, str) or not isinstance(ctl, str):
                    raise ValueError(
                        f"{self.UNS_CONTROL_MAP_KEY}['{cond}'] entries must map str treated_barcode to str control_barcode"
                    )

    def load_data(
        self,
    ) -> None:
        """
        Load the dataset and populates the perturbation truth data.
        - Validates the presence of required keys and values in `adata`:
            - `condition_key` in `adata.obs`
            - `control_name` present in `adata.obs[condition_key]`
            - `de_results_{self.deg_test_name}` in `adata.uns`
            - `control_cells_map` in `adata.uns`
        - Loads and filters differential expression results from `adata.uns`,
            keeping only genes whose differential expression meets
            user-defined thresholds.
        - Populates the `target_conditions_dict` attribute
        """
        super().load_data()
        if self.condition_key not in self.adata.obs.columns:
            raise ValueError(
                f"Condition key '{self.condition_key}' not found in adata.obs"
            )

        if not (self.adata.obs[self.condition_key] == self.control_name).any():
            raise ValueError(
                f"Data in condition key '{self.condition_key}' column does not contain control condition '{self.control_name}'"
            )

        if f"de_results_{self.deg_test_name}" not in self.adata.uns.keys():
            raise ValueError(
                f"Key 'de_results_{self.deg_test_name}' not found in adata.uns"
            )

        # Control mapping is REQUIRED: strict 1-1 map produced by preprocessing pipeline.
        if self.UNS_CONTROL_MAP_KEY not in self.adata.uns:
            raise ValueError(f"Key '{self.UNS_CONTROL_MAP_KEY}' not found in adata.uns")
        self.control_cells_ids = self.adata.uns[self.UNS_CONTROL_MAP_KEY]

        # Load and filter differential expression results
        logger.info(
            f"Loading and filtering differential expression results using {self.deg_test_name} test"
        )
        de_results_df = self.load_and_filter_deg_results()
        logger.info(f"Using {len(de_results_df)} differential expression values")

        # Optimize: Keep only necessary columns in de_results
        # Task only uses: condition_key, "gene_id", and configured metric column
        metric_column = self.de_metric_col
        necessary_columns = [self.condition_key, self.de_gene_col, metric_column]

        # TODO I think this is no longer needed. Verify using input data w/ different
        # values for de_gene_col
        # Ensure we have gene_id column for compatibility with task
        if self.de_gene_col != "gene_id":
            de_results_df = de_results_df.rename(columns={self.de_gene_col: "gene_id"})
            necessary_columns = [self.condition_key, "gene_id", metric_column]
            self.de_gene_col = "gene_id"
        de_results_df = de_results_df[necessary_columns]

        # Optional consistency checks only when mapping exists
        unique_conditions_adata = set(self.adata.obs[self.condition_key])
        if self.control_name in unique_conditions_adata:
            unique_conditions_adata.remove(self.control_name)
        unique_conditions_de_results = set(de_results_df[self.condition_key])
        if not unique_conditions_de_results.issubset(unique_conditions_adata):
            raise ValueError(
                f"de_results[{self.condition_key}] contains conditions not in adata.obs[{self.condition_key}]"
            )

        if isinstance(self.control_cells_ids, dict) and len(self.control_cells_ids) > 0:
            unique_conditions_control_cells_ids = set(self.control_cells_ids.keys())
            if unique_conditions_control_cells_ids != unique_conditions_adata:
                msg = f"Conditions in control_cells_ids and adata.obs[{self.condition_key}] are not identical"
                if unique_conditions_control_cells_ids.issubset(
                    unique_conditions_adata
                ):
                    logger.warning(
                        msg
                        + ", but control_cells_ids keys are a subset of adata.obs["
                        + self.condition_key
                        + "]. Proceeding with available mapping."
                    )
                else:
                    logger.warning(
                        msg
                        + ", and control_cells_ids contains conditions not present in adata. Task will ignore those."
                    )

        # Prepare UNS for downstream perturbation task
        self._populate_task_uns(de_results_df)

        # Validate uns contract for downstream task
        self._validate_uns_for_task(self.adata)

    @property
    def metric_column(self) -> str:
        return str(self.adata.uns.get(self.UNS_METRIC_COL_KEY, self.de_metric_col))

    @property
    def de_results(self) -> pd.DataFrame:
        value = self.adata.uns.get(self.UNS_DE_RESULTS_KEY)
        if not isinstance(value, pd.DataFrame):
            raise ValueError(
                f"adata.uns['{self.UNS_DE_RESULTS_KEY}'] must be a pandas DataFrame"
            )
        return value

    @property
    def control_mapping(self) -> Dict[str, Dict[str, List[str]]]:
        raw = self.adata.uns.get(self.UNS_CONTROL_MAP_KEY, {})
        return self._normalize_control_mapping(raw)

    @property
    def target_genes(self) -> Dict[str, List[str]]:
        value = self.adata.uns.get(self.UNS_TARGET_GENES_KEY, {})
        return value if isinstance(value, dict) else {}

    def set_control_mapping(self, raw_mapping: Dict) -> None:
        canonical = self._normalize_control_mapping(raw_mapping)
        self.adata.uns[self.UNS_CONTROL_MAP_KEY] = canonical

    def _normalize_control_mapping(
        self, raw_mapping: Dict
    ) -> Dict[str, Dict[str, List[str]]]:
        canonical: Dict[str, Dict[str, List[str]]] = {}
        if not isinstance(raw_mapping, dict):
            return {}

        obs = self.adata.obs
        for condition, mapping in raw_mapping.items():
            treated_barcodes = (
                obs.index[obs[self.condition_key] == condition].tolist()
                if self.condition_key in obs.columns
                else []
            )
            canonical_cond: Dict[str, List[str]] = {}

            if isinstance(mapping, dict):
                default_controls: List[str] = []
                if "_default" in mapping:
                    default_val = mapping["_default"]
                    default_controls = (
                        [default_val]
                        if isinstance(default_val, str)
                        else list(default_val)
                    )
                    default_controls = [str(x) for x in default_controls]

                for tb, controls in mapping.items():
                    if tb == "_default":
                        continue
                    if isinstance(controls, str):
                        ctl_list = [controls]
                    else:
                        ctl_list = list(controls)
                    canonical_cond[str(tb)] = [str(x) for x in ctl_list]

                for tb in treated_barcodes:
                    if tb not in canonical_cond and default_controls:
                        canonical_cond[str(tb)] = list(default_controls)

            elif isinstance(mapping, str) or isinstance(
                mapping, (list, tuple, np.ndarray)
            ):
                default_controls = (
                    [mapping] if isinstance(mapping, str) else list(mapping)
                )
                default_controls = [str(x) for x in default_controls]
                for tb in treated_barcodes:
                    canonical_cond[str(tb)] = list(default_controls)
            else:
                continue

            for tb in canonical_cond.keys():
                vals = canonical_cond[tb]
                canonical_cond[tb] = [
                    str(x) for x in (vals if isinstance(vals, list) else [vals])
                ]

            canonical[condition] = canonical_cond

        return canonical

    def get_controls(
        self, condition: str, treated_barcode: Optional[str] = None
    ) -> List[str]:
        mapping = self.control_mapping
        if condition not in mapping:
            raise ValueError(f"Condition {condition} not found in control mapping")

        # If a specific treated barcode is provided and known, return its controls
        if treated_barcode is not None and treated_barcode in mapping[condition]:
            return list(mapping[condition][treated_barcode])

        # Otherwise, return the union of all control barcodes for the condition
        control_union: set[str] = set()
        for ctl_list in mapping[condition].values():
            control_union.update(list(ctl_list))
        return sorted(control_union)

    def get_indices_for(
        self, condition: str, treated_barcodes: Optional[List[str]] = None
    ) -> tuple[np.ndarray, np.ndarray]:
        obs = self.adata.obs
        obs_index = obs.index

        if treated_barcodes is None:
            treated_barcodes = obs_index[obs[self.condition_key] == condition].tolist()

        control_union: set[str] = set()
        for tb in treated_barcodes:
            control_union.update(self.get_controls(condition, tb))

        treated_rows = obs_index.get_indexer_for(treated_barcodes)
        control_rows = obs_index.get_indexer_for(list(control_union))
        treated_rows = treated_rows[treated_rows >= 0]
        control_rows = control_rows[control_rows >= 0]
        return treated_rows, control_rows

    def store_task_inputs(self) -> Path:
        """
        Store all task inputs into a single .h5ad file.

        The AnnData object contains in uns:
        - target_conditions_dict
        - de_results (DataFrame with required columns)
        - control_cells_ids

        Returns:
            Path: Path to the task inputs directory.
        """
        # Ensure the task inputs directory exists
        self.task_inputs_dir.mkdir(parents=True, exist_ok=True)
        adata_file = self.task_inputs_dir / "perturbation_task_inputs.h5ad"
        # Write single AnnData with embedded artifacts
        self.adata.write_h5ad(adata_file)
        return self.task_inputs_dir

    def _validate(self) -> None:
        """
        Perform dataset-specific validation.

        Validates the following:
        - Ensures that ``condition_key`` exists in ``adata.obs``.
        - Ensures that each condition in ``adata.obs[condition_key]`` is present
            in the control mapping or is the control name.
        - Ensures that every condition present in the differential expression
            results or in the target conditions dict is present in the control mapping.

        Notes:
        - This method does not strictly enforce condition label formatting and does
            not explicitly validate combinatorial perturbations.

        Raises:
            ValueError: If required keys or mappings are missing from adata.obs or adata.uns.
        """
        super()._validate()

        if self.condition_key not in self.adata.obs.columns:
            raise ValueError(
                f"Condition key '{self.condition_key}' not found in adata.obs"
            )
        # Validate conditions found in the original adata
        original_conditions = set(self.adata.obs[self.condition_key])
        mapped_conditions = set(self.control_mapping.keys())
        for condition in original_conditions:
            # Strict schema on format, but allow extra perturbations not in DE with a warning
            if condition in mapped_conditions:
                continue
            elif condition == self.control_name:
                continue
            else:
                logger.warning(
                    f"Unexpected condition label: {condition} not present in control mapping."
                )
                continue

        target_conditions = set(getattr(self, "target_conditions_dict", {}).keys())
        # Also allow any condition that appears in de_results (some datasets may include
        # conditions beyond the sampled target set)
        try:
            de_res_obj = None
            # Prefer the standardized key populated by this dataset when present
            if self.UNS_DE_RESULTS_KEY in self.adata.uns:
                de_res_obj = self.adata.uns.get(self.UNS_DE_RESULTS_KEY)

            de_res_df = (
                pd.DataFrame(de_res_obj) if isinstance(de_res_obj, dict) else de_res_obj
            )
            if (
                isinstance(de_res_df, pd.DataFrame)
                and self.condition_key in de_res_df.columns
            ):
                target_conditions = target_conditions.union(
                    set(de_res_df[self.condition_key].astype(str).unique())
                )
        except Exception:
            logger.warning("No differential expression results found in adata.uns")

        for condition in target_conditions:
            # Strict schema on format, but allow extra perturbations not in DE with a warning
            if condition not in mapped_conditions:
                logger.warning(
                    f"Unexpected condition label: {condition}."
                    f"not present in control mapping."
                )
                continue
