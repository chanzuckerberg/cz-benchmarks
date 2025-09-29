import logging
from typing import Dict, List, Literal, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp_sparse

from ...constants import RANDOM_SEED
from ...metrics import metrics_registry
from ...metrics.types import MetricResult, MetricType
from ...tasks.types import CellRepresentation
from ..task import Task, TaskInput, TaskOutput
from ..utils import looks_like_lognorm

logger = logging.getLogger(__name__)


class PerturbationExpressionPredictionTaskInput(TaskInput):
    """Pydantic model for Perturbation task inputs.

    Optionally carries the predictions' ordering via cell_index/gene_index so the
    task can align a model matrix that is a subset or re-ordered relative to
    the dataset adata.
    """

    adata: ad.AnnData
    gene_index: Optional[pd.Index] = None
    cell_index: Optional[pd.Index] = None


def build_task_input_from_predictions(
    predictions_adata: ad.AnnData,
    dataset_adata: ad.AnnData,
) -> PerturbationExpressionPredictionTaskInput:
    """Create a task input from a predictions AnnData and the dataset AnnData.

    This preserves the predictions' obs/var order so the task can align matrices
    without forcing the caller to reorder arrays.
    """
    return PerturbationExpressionPredictionTaskInput(
        adata=dataset_adata,
        gene_index=predictions_adata.var.index,
        cell_index=predictions_adata.obs.index,
    )


class PerturbationExpressionPredictionOutput(TaskOutput):
    """Output for perturbation task."""

    pred_log_fc_dict: Dict[str, np.ndarray]
    true_log_fc_dict: Dict[str, np.ndarray]


class PerturbationExpressionPredictionTask(Task):
    display_name = "Perturbation Expression Prediction"
    description = "Evaluate the quality of predicted changes in expression levels for genes that are differentially expressed under perturbation(s) using multiple classification and correlation metrics."
    input_model = PerturbationExpressionPredictionTaskInput

    def __init__(
        self,
        condition_key: str = "condition",
        control_name: str = "ctrl",
        *,
        pred_effect_operation: Literal["difference", "ratio"] = "ratio",
        random_seed: int = RANDOM_SEED,
    ):
        """
        Args:
            condition_key (str): Key for the column in `adata.obs` specifying
                conditions. Defaults to "condition".
            control_name (str): Prefix for control conditions. Defaults to "ctrl".
            pred_effect_operation (Literal["difference", "ratio"]): How to compute predicted
                effect between treated and control mean predictions over genes. "difference"
                uses mean(treated) - mean(control) and is generally safe across scales
                (probabilities, z-scores, raw expression). "ratio" uses log((mean(treated)+eps)/(mean(control)+eps))
                when means are positive; if non-positive values are detected it falls back to "difference".
            random_seed (int): Random seed for reproducibility.
        """
        super().__init__(random_seed=random_seed)
        self.control_name = control_name
        self.condition_key = condition_key
        self.pred_effect_operation = pred_effect_operation

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: PerturbationExpressionPredictionTaskInput,
    ) -> PerturbationExpressionPredictionOutput:
        """
        Runs the perturbation evaluation task.

        Args:
            cell_representation: Cell expression matrix of shape (n_cells, n_genes)
            task_input: Task input containing AnnData with all necessary data

        Returns:
            PerturbationExpressionPredictionOutput: Predicted and true log fold changes
        """
        self._validate(task_input, cell_representation)
        # Detect if predictions are already log-normalized; computation will adapt accordingly
        is_lognorm_predictions = looks_like_lognorm(cell_representation)
        pred_log_fc_dict: Dict[str, np.ndarray] = {}
        true_log_fc_dict: Dict[str, np.ndarray] = {}
        adata = task_input.adata

        obs = adata.obs
        obs_index = obs.index
        var_index = adata.var.index

        # Predictions index spaces; default to dataset order if not provided
        pred_cell_index = (
            task_input.cell_index if task_input.cell_index is not None else obs_index
        )
        pred_gene_index = (
            task_input.gene_index if task_input.gene_index is not None else var_index
        )

        de_results: pd.DataFrame = adata.uns["de_results"]
        metric_column: str = adata.uns.get("metric_column", "logfoldchange")
        # Strict 1-1 mapping is required
        control_map_1to1: Optional[Dict] = adata.uns.get("control_cells_map")
        if not isinstance(control_map_1to1, dict):
            raise ValueError(
                "adata.uns['control_cells_map'] is required and must be a dict of treated->control mappings per condition."
            )
        target_conditions_dict: Dict[str, List[str]] = adata.uns.get(
            "target_conditions_dict", {}
        )

        perturbation_conditions = de_results[self.condition_key].unique().tolist()

        for condition in perturbation_conditions:
            # Select genes for this condition
            condition_de = de_results[de_results[self.condition_key] == condition]
            if (
                condition in target_conditions_dict
                and target_conditions_dict[condition]
            ):
                candidate_genes = [
                    g
                    for g in target_conditions_dict[condition]
                    if g in set(condition_de["gene_id"].values)
                ]
            else:
                # Skip conditions that don't have target conditions defined
                continue

            if len(candidate_genes) == 0:
                continue

            # Ground truth vector
            true_lfc_series = condition_de.set_index("gene_id").reindex(
                candidate_genes
            )[metric_column]
            true_lfc = true_lfc_series.values
            valid_mask = ~np.isnan(true_lfc)
            if not valid_mask.any():
                continue
            genes = np.asarray(candidate_genes)[valid_mask]
            true_lfc = true_lfc[valid_mask]

            # Map genes to predictions' columns
            gene_idx = pred_gene_index.get_indexer(genes)
            keep = gene_idx >= 0
            if not keep.any():
                continue
            genes = genes[keep]
            true_lfc = true_lfc[keep]
            gene_idx = gene_idx[keep]

            # Resolve treated and control barcodes
            treated_barcodes: List[str] = []
            control_barcodes: List[str] = []

            # Compute per-pair differences using the strict 1-1 map
            if condition not in control_map_1to1 or not isinstance(
                control_map_1to1[condition], dict
            ):
                raise ValueError(
                    f"Missing 1-1 control mapping for condition '{condition}' in adata.uns['control_cells_map']"
                )

            mapping: Dict[str, str] = control_map_1to1[condition]  # treated -> control
            treated_rows: List[int] = []
            control_rows: List[int] = []
            for tb, ctl in mapping.items():
                tb_idx = pred_cell_index.get_indexer_for([str(tb)])
                ctl_idx = pred_cell_index.get_indexer_for([str(ctl)])
                if tb_idx.size == 0 or ctl_idx.size == 0:
                    continue
                tb_row = tb_idx[0]
                ctl_row = ctl_idx[0]
                if tb_row < 0 or ctl_row < 0:
                    continue
                treated_rows.append(int(tb_row))
                control_rows.append(int(ctl_row))

            if len(treated_rows) == 0:
                continue

            # Compute mean prediction per group (treated vs control) for the selected genes
            treated_matrix = cell_representation[np.ix_(treated_rows, gene_idx)]
            control_matrix = cell_representation[np.ix_(control_rows, gene_idx)]

            if sp_sparse.issparse(treated_matrix):
                treated_matrix = treated_matrix.toarray()
            if sp_sparse.issparse(control_matrix):
                control_matrix = control_matrix.toarray()

            treated_mean = np.mean(treated_matrix, axis=0)
            control_mean = np.mean(control_matrix, axis=0)

            # Compute predicted log fold-change depending on configuration and scale
            eps = 1e-8
            if self.pred_effect_operation == "difference":
                # Use difference regardless of scale; this is safest for z-scores and bounded scores
                pred_lfc = np.asarray(treated_mean - control_mean).ravel()
            else:  # "ratio"
                if is_lognorm_predictions:
                    # If already log scale, ratio corresponds to difference
                    pred_lfc = np.asarray(treated_mean - control_mean).ravel()
                else:
                    # Raw scale ratio; guard against non-positive means by falling back to difference
                    if np.any(treated_mean <= 0.0) or np.any(control_mean <= 0.0):
                        pred_lfc = np.asarray(treated_mean - control_mean).ravel()
                    else:
                        pred_lfc = np.log(
                            (treated_mean + eps) / (control_mean + eps)
                        ).ravel()
            pred_log_fc_dict[condition] = np.asarray(pred_lfc).ravel()
            true_log_fc_dict[condition] = np.asarray(true_lfc).ravel()

        return PerturbationExpressionPredictionOutput(
            pred_log_fc_dict=pred_log_fc_dict,
            true_log_fc_dict=true_log_fc_dict,
        )

    def _compute_metrics(
        self,
        task_input: PerturbationExpressionPredictionTaskInput,
        task_output: PerturbationExpressionPredictionOutput,
    ) -> List[MetricResult]:
        """
        Compute perturbation prediction quality using Spearman rank correlation
        between predicted and true log fold changes for each condition.
        """
        spearman_correlation_metric = MetricType.SPEARMAN_CORRELATION_CALCULATION

        metric_results: List[MetricResult] = []
        for condition in task_output.pred_log_fc_dict.keys():
            pred_log_fc = task_output.pred_log_fc_dict[condition]
            true_log_fc = task_output.true_log_fc_dict[condition]

            spearman_corr = metrics_registry.compute(
                spearman_correlation_metric,
                a=true_log_fc,
                b=pred_log_fc,
            )
            spearman_corr_value = getattr(spearman_corr, "correlation", spearman_corr)
            metric_results.append(
                MetricResult(
                    metric_type=spearman_correlation_metric,
                    value=spearman_corr_value,
                    params={"condition": condition},
                )
            )
        return metric_results

    @staticmethod
    def compute_baseline(
        cell_representation: CellRepresentation,
        baseline_type: Literal["median", "mean"] = "median",
    ) -> CellRepresentation:
        """Set a baseline perturbation prediction using mean or median expression.

        This method creates a baseline prediction by either taking the mean or
        the median of the control cells' gene expression. This baseline
        represents a simple no-change prediction.

        Args:
            cell_representation: The gene expression matrix of control cells.
            baseline_type: The type of baseline to use, either "mean" or "median".

        Returns:
            A DataFrame representing the baseline perturbation prediction.
        """
        # Create baseline prediction by replicating the aggregated expression values
        # across all cells in the dataset.
        baseline_func = np.median if baseline_type == "median" else np.mean
        if baseline_type == "median" and sp_sparse.issparse(cell_representation):
            cell_representation = cell_representation.toarray()

        perturb_baseline_pred = np.tile(
            baseline_func(cell_representation, axis=0),
            (cell_representation.shape[0], 1),
        )

        # Store the baseline prediction in the dataset for evaluation
        return perturb_baseline_pred

    def _validate(
        self,
        task_input: PerturbationExpressionPredictionTaskInput,
        cell_representation: CellRepresentation,
    ) -> None:
        # Allow both log-normalized and raw predictions. Downstream computation adapts accordingly.

        adata = task_input.adata
        # Allow callers to pass predictions with custom ordering/subsets via indices.
        # If indices are not provided, enforce exact shape equality with adata.
        has_custom_ordering = hasattr(task_input, "cell_index") or hasattr(task_input, "gene_index")
            getattr(task_input, "cell_index", None) is not None
            or getattr(task_input, "gene_index", None) is not None
        )
        if not has_custom_ordering:
            if cell_representation.shape != (adata.n_obs, adata.n_vars):
                raise ValueError(
                    "Predictions must match adata shape (n_obs, n_vars) when no indices are provided."
                )
        else:
            # Basic dimensionality checks when indices are supplied
            if task_input.cell_index is not None and cell_representation.shape[
                0
            ] != len(task_input.cell_index):
                raise ValueError(
                    "Number of prediction rows must match length of provided cell_index."
                )
            if task_input.gene_index is not None and cell_representation.shape[
                1
            ] != len(task_input.gene_index):
                raise ValueError(
                    "Number of prediction columns must match length of provided gene_index."
                )

        if "de_results" not in adata.uns:
            raise ValueError("adata.uns['de_results'] is required.")
        de_results = adata.uns["de_results"]
        if not isinstance(de_results, pd.DataFrame):
            raise ValueError("adata.uns['de_results'] must be a pandas DataFrame.")

        metric_column = adata.uns.get("metric_column", "logfoldchange")
        for col in [self.condition_key, "gene_id", metric_column]:
            if col not in de_results.columns:
                raise ValueError(f"de_results missing required column '{col}'")

        cm = adata.uns.get("control_cells_map")
        if not isinstance(cm, dict):
            raise ValueError(
                "adata.uns['control_cells_map'] is required and must be a dict."
            )
