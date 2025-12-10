"""
Rare cell type detection task for evaluating model embeddings.

This task evaluates how well cell representation models can identify and classify
rare cell types by training classifiers on embeddings and measuring performance
on cell types that constitute a small fraction of the total population.
"""

import logging
from typing import Annotated, List, Dict, Any, Optional

import numpy as np
import pandas as pd
from pydantic import Field, field_validator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from ...constants import RANDOM_SEED
from ..constants import N_FOLDS
from ..task import NoBaselineInput, Task, TaskInput, TaskOutput
from ...tasks.types import CellRepresentation
from ...types import ListLike
from ...metrics.types import MetricResult, MetricType

logger = logging.getLogger(__name__)


class KkTestTaskInput(TaskInput):
    """Pydantic model for KkTestTask inputs."""

    obs: Annotated[
        str,
        Field(
            description="Column name in AnnData.obs containing cell type labels."
        ),
    ]
    rarity_threshold: Annotated[
        float,
        Field(
            description="Maximum frequency threshold for considering a cell type as rare (0-1).",
            gt=0.0,
            le=1.0,
        ),
    ] = 0.05
    min_cells: Annotated[
        int,
        Field(
            description="Minimum number of cells required for a cell type to be included in evaluation.",
            ge=1,
        ),
    ] = 10
    n_splits: Annotated[
        int,
        Field(
            description="Number of cross-validation folds.",
            ge=2,
        ),
    ] = N_FOLDS

    @field_validator("obs")
    @classmethod
    def _validate_obs(cls, v: str) -> str:
        if not isinstance(v, str) or len(v) == 0:
            raise ValueError("obs must be a non-empty string.")
        return v



class KkTestTaskOutput(TaskOutput):
    """Output model for KkTestTask."""

    rare_types: List[str]
    classifier_results: List[Dict[str, Any]]
    n_rare_cells: int
    n_common_cells: int
    total_cells: int


class KkTestTask(Task):
    """Task for evaluating rare cell type detection performance.

    This task evaluates how well cell embeddings can distinguish rare cell types
    from common ones. It identifies rare cell types based on frequency thresholds,
    then trains multiple classifiers (Logistic Regression, KNN, Random Forest) to
    predict cell types. Performance is measured using cross-validation with metrics
    including F1 score, precision, recall, balanced accuracy, and MCC.

    The task focuses on detecting rare cell types, which are often critical in
    biological contexts (e.g., rare immune cell subtypes, stem cell populations).
    """

    display_name = "Kk Test Task"
    description = "Evaluate rare cell type detection performance using multiple classifiers."

    input_model = KkTestTaskInput
    baseline_model = NoBaselineInput

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)

    def _identify_rare_types(
        self,
        labels: pd.Series,
        rarity_threshold: float,
        min_cells: int,
    ) -> List[str]:
        """Identify rare cell types based on frequency threshold.

        Args:
            labels: Cell type labels
            rarity_threshold: Maximum frequency for rare types (0-1)
            min_cells: Minimum number of cells required

        Returns:
            List of rare cell type names
        """
        counts = labels.value_counts()
        freqs = counts / len(labels)

        rare_types = freqs[
            (freqs <= rarity_threshold) & (counts >= min_cells)
        ].index.tolist()

        logger.info(f"Found {len(rare_types)} rare cell types:")
        for ct in rare_types:
            logger.info(f"  {ct}: {counts[ct]} cells ({freqs[ct]*100:.2f}%)")

        return rare_types

    def _compute_binary_metrics(
        self, y_true_rare: np.ndarray, y_pred_rare: np.ndarray
    ) -> Dict[str, float]:
        """Compute binary classification metrics for rare vs non-rare.

        Args:
            y_true_rare: Binary array indicating true rare cells
            y_pred_rare: Binary array indicating predicted rare cells

        Returns:
            Dictionary of metric values
        """
        metrics = {
            "precision": precision_score(y_true_rare, y_pred_rare, zero_division=0),
            "recall": recall_score(y_true_rare, y_pred_rare, zero_division=0),
            "f1": f1_score(y_true_rare, y_pred_rare, zero_division=0),
            "balanced_acc": balanced_accuracy_score(y_true_rare, y_pred_rare),
        }

        # MCC - good for imbalanced data
        tp = (y_true_rare & y_pred_rare).sum()
        tn = ((~y_true_rare) & (~y_pred_rare)).sum()
        fp = ((~y_true_rare) & y_pred_rare).sum()
        fn = (y_true_rare & ~y_pred_rare).sum()

        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        metrics["mcc"] = mcc_num / mcc_den if mcc_den > 0 else 0

        # Specificity
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

        return metrics

    def _compute_per_type_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, rare_types: List[str]
    ) -> Dict[str, float]:
        """Compute per-type metrics for each individual rare type.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            rare_types: List of rare type names

        Returns:
            Dictionary of aggregated per-type metrics
        """
        recalls = []
        precisions = []

        for ct in rare_types:
            true_mask = y_true == ct
            pred_mask = y_pred == ct

            if true_mask.sum() == 0:
                continue

            tp = (true_mask & pred_mask).sum()
            fp = (~true_mask & pred_mask).sum()
            fn = (true_mask & ~pred_mask).sum()

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0

            recalls.append(recall)
            precisions.append(precision)

        return {
            "mean_recall_per_type": np.mean(recalls) if recalls else 0,
            "min_recall_per_type": np.min(recalls) if recalls else 0,
            "mean_precision_per_type": np.mean(precisions) if precisions else 0,
        }

    def _eval_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        rare_types: List[str],
        clf,
    ) -> Dict[str, float]:
        """Evaluate a single cross-validation fold.

        Args:
            X_train: Training embeddings
            X_test: Test embeddings
            y_train: Training labels
            y_test: Test labels
            rare_types: List of rare cell type names
            clf: Classifier instance

        Returns:
            Dictionary of metrics for this fold
        """
        # Train
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Convert to binary (rare vs not)
        y_true_rare = np.isin(y_test, rare_types)
        y_pred_rare = np.isin(y_pred, rare_types)

        # Compute metrics
        metrics = {}
        metrics.update(self._compute_binary_metrics(y_true_rare, y_pred_rare))
        metrics.update(self._compute_per_type_metrics(y_test, y_pred, rare_types))

        return metrics

    def _run_cv_eval(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        rare_types: List[str],
        clf,
        n_splits: int,
    ) -> Dict[str, float]:
        """Run stratified k-fold cross-validation.

        Args:
            embeddings: Cell embeddings
            labels: Cell type labels
            rare_types: List of rare cell type names
            clf: Classifier instance
            n_splits: Number of CV folds

        Returns:
            Dictionary of averaged metrics across folds
        """
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_seed
        )

        fold_results = []
        for train_idx, test_idx in skf.split(embeddings, labels):
            X_train, X_test = embeddings[train_idx], embeddings[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            fold_metrics = self._eval_fold(
                X_train, X_test, y_train, y_test, rare_types, clf
            )
            fold_results.append(fold_metrics)

        # Average across folds
        avg_metrics = {}
        for key in fold_results[0].keys():
            values = [f[key] for f in fold_results]
            avg_metrics[key] = np.mean(values)
            avg_metrics[f"{key}_std"] = np.std(values)

        return avg_metrics

    def _run_task(
        self,
        cell_representation: CellRepresentation,
        task_input: KkTestTaskInput,
    ) -> KkTestTaskOutput:
        """Run rare cell type detection evaluation.

        Args:
            cell_representation: Cell embeddings (can be AnnData or array)
            task_input: Task configuration parameters

        Returns:
            KkTestTaskOutput with results and intermediate data
        """
        # Extract embeddings and labels
        if hasattr(cell_representation, "obs"):
            # AnnData object
            # Handle both dense and sparse matrices
            if hasattr(cell_representation.X, "toarray"):
                # Sparse matrix
                embeddings = cell_representation.X.toarray()
            else:
                # Dense matrix/array
                embeddings = np.array(cell_representation.X)

            if task_input.obs not in cell_representation.obs.columns:
                raise ValueError(
                    f"Column '{task_input.obs}' not found in AnnData.obs. "
                    f"Available columns: {cell_representation.obs.columns.tolist()}"
                )
            labels = pd.Series(cell_representation.obs[task_input.obs].values)
        else:
            # Assume it's an array-like
            embeddings = np.array(cell_representation)
            raise ValueError(
                "cell_representation must be an AnnData object with .obs attribute"
            )

        logger.info(
            f"Processing data: {embeddings.shape[0]} cells, {embeddings.shape[1]} features"
        )

        # Find rare types
        rare_types = self._identify_rare_types(
            labels, task_input.rarity_threshold, task_input.min_cells
        )

        if len(rare_types) == 0:
            logger.warning(
                "No rare types found! Try adjusting rarity_threshold or min_cells"
            )
            # Return empty results
            return KkTestTaskOutput(
                rare_types=[],
                classifier_results=[],
                n_rare_cells=0,
                n_common_cells=len(labels),
                total_cells=len(labels),
            )

        # Convert labels to array
        labels_array = labels.values

        # Count rare vs common cells
        n_rare_cells = np.isin(labels_array, rare_types).sum()
        n_common_cells = len(labels) - n_rare_cells

        logger.info(
            f"Rare cells: {n_rare_cells}, Common cells: {n_common_cells}, "
            f"Total: {len(labels)}"
        )

        # Define classifiers
        classifiers = {
            "logistic": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=self.random_seed
            ),
            "knn": KNeighborsClassifier(n_neighbors=15),
            "rf": RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                max_depth=20,
                random_state=self.random_seed,
                n_jobs=-1,
            ),
        }

        # Run evaluation for each classifier
        results = []
        for name, clf in classifiers.items():
            logger.info(f"Running {name} classifier...")
            metrics = self._run_cv_eval(
                embeddings, labels_array, rare_types, clf, task_input.n_splits
            )
            metrics["classifier"] = name
            results.append(metrics)

        # Store results for metrics computation
        self._classifier_results = results
        self._rare_types = rare_types

        return KkTestTaskOutput(
            rare_types=rare_types,
            classifier_results=results,
            n_rare_cells=int(n_rare_cells),
            n_common_cells=int(n_common_cells),
            total_cells=len(labels),
        )

    def _compute_metrics(
        self,
        task_input: KkTestTaskInput,
        task_output: KkTestTaskOutput,
    ) -> List[MetricResult]:
        """Compute metrics from task output.

        Args:
            task_input: Task input configuration
            task_output: Task output containing results

        Returns:
            List of MetricResult objects
        """
        logger.info("Computing rare cell detection metrics...")

        if len(task_output.classifier_results) == 0:
            logger.warning("No results to compute metrics from")
            return []

        metrics_list = []

        # Create results dataframe
        results_df = pd.DataFrame(task_output.classifier_results)

        # Define metric mappings (using mean fold metrics since we use CV)
        metric_mappings = {
            "f1": MetricType.MEAN_FOLD_F1_SCORE,
            "precision": MetricType.MEAN_FOLD_PRECISION,
            "recall": MetricType.MEAN_FOLD_RECALL,
            "balanced_acc": MetricType.MEAN_FOLD_ACCURACY,  # Using balanced accuracy
        }

        # Add metrics for mean across all classifiers
        base_params = {
            "classifier": "MEAN(logistic,knn,rf)",
            "n_rare_types": len(task_output.rare_types),
        }

        for metric_name, metric_type in metric_mappings.items():
            if metric_name in results_df.columns:
                metrics_list.append(
                    MetricResult(
                        metric_type=metric_type,
                        value=float(results_df[metric_name].mean()),
                        params=base_params,
                    )
                )

        # Add per-classifier metrics
        for clf in results_df["classifier"].unique():
            clf_df = results_df[results_df["classifier"] == clf]
            clf_params = {
                "classifier": clf,
                "n_rare_types": len(task_output.rare_types),
            }

            for metric_name, metric_type in metric_mappings.items():
                if metric_name in clf_df.columns:
                    metrics_list.append(
                        MetricResult(
                            metric_type=metric_type,
                            value=float(clf_df[metric_name].iloc[0]),
                            params=clf_params,
                        )
                    )

        logger.info(f"Computed {len(metrics_list)} metrics")
        return metrics_list

    def compute_baseline(
        self,
        expression_data: CellRepresentation,
        baseline_input: NoBaselineInput = None,
    ):
        """Compute baseline for rare cell detection.

        Not implemented as there is no standard baseline for this task.
        """
        raise NotImplementedError(
            "Baseline not implemented for rare cell detection task"
        )
