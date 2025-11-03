import itertools
import logging
from typing import Annotated, List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
from pydantic import Field, field_validator
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ...constants import RANDOM_SEED
from ..constants import N_FOLDS
from ..task import NoBaselineInput, Task, TaskInput, TaskOutput
from ...tasks.types import CellRepresentation
from ...types import ListLike
from ...metrics.types import MetricResult, MetricType
from ...datasets.types import Organism
from ..utils import aggregate_cells_to_samples


logger = logging.getLogger(__name__)


class CrossSpeciesLabelPredictionTaskInput(TaskInput):
    """Pydantic model for CrossSpeciesLabelPredictionTask inputs."""

    labels: Annotated[
        List[ListLike],
        Field(
            description="List of ground truth labels for each species dataset (e.g., cell types)."
        ),
    ]
    organisms: Annotated[
        List[Organism],
        Field(
            description="List of organisms corresponding to each dataset for cross-species evaluation."
        ),
    ]
    sample_ids: Annotated[
        Optional[List[ListLike]],
        Field(
            description="Optional list of sample/donor IDs for aggregation, one per dataset."
        ),
    ] = None
    aggregation_method: Annotated[
        Literal["none", "mean", "median"],
        Field(
            description="Method to aggregate cells with the same sample_id ('none', 'mean', or 'median')."
        ),
    ] = "mean"
    n_folds: Annotated[
        int,
        Field(
            description="Number of cross-validation folds for intra-species evaluation."
        ),
    ] = N_FOLDS

    @field_validator("organisms")
    @classmethod
    def _validate_organisms(cls, v: List[Organism]) -> List[Organism]:
        if not isinstance(v, list):
            raise ValueError("organisms must be a list of organisms.")
        return v

    @field_validator("sample_ids")
    @classmethod
    def _validate_sample_ids(
        cls, v: Optional[List[ListLike]]
    ) -> Optional[List[ListLike]]:
        if v is not None and not isinstance(v, list):
            raise ValueError("sample_ids must be a list of list-like objects.")
        return v

    @field_validator("n_folds")
    @classmethod
    def _validate_n_folds(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("n_folds must be a positive integer.")
        return v


class CrossSpeciesLabelPredictionOutput(TaskOutput):
    results: List[Dict[str, Any]]  # List of dicts with classifier, split, and metrics


class CrossSpeciesLabelPredictionTask(Task):
    """Task for cross-species label prediction evaluation.

    This task evaluates cross-species transfer by training classifiers on one species
    and testing on another species. It computes accuracy, F1, precision, recall, and AUROC
    for multiple classifiers (Logistic Regression, KNN, Random Forest).

    The task can optionally aggregate cell-level embeddings to sample/donor level
    before running classification.
    """

    display_name = "cross-species label prediction"
    description = "Evaluate cross-species label prediction performance using multiple classifiers."

    input_model = CrossSpeciesLabelPredictionTaskInput
    baseline_model = NoBaselineInput

    def __init__(self, *, random_seed: int = RANDOM_SEED):
        super().__init__(random_seed=random_seed)
        self.requires_multiple_datasets = True

    def _run_cross_species_classification(
        self,
        train_embeddings: np.ndarray,
        train_labels: pd.Series,
        test_embeddings: np.ndarray,
        test_labels: pd.Series,
        train_species: str,
        test_species: str,
    ) -> List[Dict[str, Any]]:
        """Run cross-species classification for multiple classifiers.

        Args:
            train_embeddings: Training embeddings
            train_labels: Training labels
            test_embeddings: Test embeddings
            test_labels: Test labels
            train_species: Name of training species
            test_species: Name of test species

        Returns:
            List of result dictionaries with metrics for each classifier
        """
        train_labels_cat = pd.Categorical(train_labels.astype(str))
        test_labels_cat = pd.Categorical(
            test_labels.astype(str), categories=train_labels_cat.categories
        )

        train_label_codes = train_labels_cat.codes
        test_label_codes = test_labels_cat.codes

        n_classes = len(train_labels_cat.categories)
        target_type = "binary" if n_classes == 2 else "macro"

        logger.info(
            f"Cross-species classification: {train_species} -> {test_species}, "
            f"{n_classes} classes, using {target_type} averaging"
        )

        scorers = {
            "accuracy": make_scorer(accuracy_score),
            "f1": make_scorer(f1_score, average=target_type),
            "precision": make_scorer(precision_score, average=target_type),
            "recall": make_scorer(recall_score, average=target_type),
            "auroc": make_scorer(
                roc_auc_score,
                average="macro",
                multi_class="ovr",
                response_method="predict_proba",
            ),
        }

        classifiers = {
            "lr": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=1000, random_state=self.random_seed
                        ),
                    ),
                ]
            ),
            "knn": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
            "rf": Pipeline(
                [
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=100, random_state=self.random_seed
                        ),
                    )
                ]
            ),
        }

        results = []

        for name, clf in classifiers.items():
            logger.info(f"Training {name} classifier...")
            clf.fit(train_embeddings, train_label_codes)

            clf_results = {
                "classifier": name,
                "train_species": train_species,
                "test_species": test_species,
            }

            for metric_name, scorer in scorers.items():
                clf_results[metric_name] = scorer(
                    clf, test_embeddings, test_label_codes
                )

            results.append(clf_results)
            logger.debug(f"{name} results: {clf_results}")

        return results

    def _run_cross_validation_classification(
        self,
        embeddings: np.ndarray,
        labels: pd.Series,
        species: str,
        sample_ids: Optional[pd.Series] = None,
        n_folds: int = N_FOLDS,
    ) -> List[Dict[str, Any]]:
        """Run stratified cross-validation for multiple classifiers.

        Args:
            embeddings: embeddings
            sample_ids: donor or sample identifiers, used to balance the folds
            labels: labels
            species: name of species

        Returns:
            List of result dictionaries with metrics for each classifier
        """
        if sample_ids is None:
            selector = StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_seed
            )
        else:
            # we need to use StratifiedGroupKFold so the sample_ids in train/test are completely disjoint
            selector = StratifiedGroupKFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_seed
            )

        labels = pd.Categorical(labels.astype(str))
        label_codes = labels.codes

        n_classes = len(labels.categories)
        target_type = "binary" if n_classes == 2 else "macro"

        logger.info(
            f"Cross-validated classification: {species}, "
            f"{n_classes} classes, using {target_type} averaging"
        )

        scorers = {
            "accuracy": make_scorer(accuracy_score),
            "f1": make_scorer(f1_score, average=target_type),
            "precision": make_scorer(precision_score, average=target_type),
            "recall": make_scorer(recall_score, average=target_type),
            "auroc": make_scorer(
                roc_auc_score,
                average="macro",
                multi_class="ovr",
                response_method="predict_proba",
            ),
        }

        classifiers = {
            "lr": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=1000, random_state=self.random_seed
                        ),
                    ),
                ]
            ),
            "knn": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("knn", KNeighborsClassifier(n_neighbors=5)),
                ]
            ),
            "rf": Pipeline(
                [
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=100, random_state=self.random_seed
                        ),
                    )
                ]
            ),
        }

        results = []

        for name, clf in classifiers.items():
            for fold_idx, (train_idx, test_idx) in enumerate(
                selector.split(embeddings, label_codes, groups=sample_ids)
            ):
                train_emb, test_emb = embeddings[train_idx], embeddings[test_idx]
                train_labels, test_labels = (
                    label_codes[train_idx],
                    label_codes[test_idx],
                )

                logger.info(f"Training {name} classifier...")
                clf.fit(train_emb, train_labels)

                fold_results = {
                    "classifier": name,
                    "split": fold_idx,
                    "train_species": species,
                    "test_species": species,
                }

                for metric_name, scorer in scorers.items():
                    fold_results[metric_name] = scorer(clf, test_emb, test_labels)

                results.append(fold_results)
                logger.debug(f"{name}, fold {fold_idx} results: {fold_results}")

        return results

    def _run_task(
        self,
        cell_representation: List[CellRepresentation],
        task_input: CrossSpeciesLabelPredictionTaskInput,
    ) -> CrossSpeciesLabelPredictionOutput:
        """Run cross-species label prediction evaluation.

        Args:
            cell_representation: List of cell representations for each species
            task_input: Task input containing labels and organism information

        Returns:
            CrossSpeciesLabelPredictionOutput: Results from cross-species evaluation
        """
        logger.debug(
            f"CrossSpeciesLabelPredictionTask._run_task: cell_representation type={type(cell_representation)}, "
            f"n_datasets={len(cell_representation) if isinstance(cell_representation, list) else 1}"
        )
        if task_input.sample_ids is None:
            task_input.sample_ids = [None for _ in cell_representation]

        lengths = {
            len(cell_representation),
            len(task_input.organisms),
            len(task_input.labels),
            len(task_input.sample_ids),
        }
        logger.debug(
            "CrossSpeciesLabelPredictionTask._run_task: "
            "len(cell_representation)=%d, len(task_input.organisms)=%d, "
            "len(task_input.labels)=%d, len(task_input.sample_ids)=%d, len(lengths)=%d",
            len(cell_representation),
            len(task_input.organisms),
            len(task_input.labels),
            len(task_input.sample_ids),
            len(lengths),
        )
        if len(lengths) != 1:
            raise ValueError(
                f"Number of cell representations ({len(cell_representation)}) must match "
                f"number of items in the task inputs "
                f"(got {len(task_input.organisms)} organisms, {len(task_input.labels)} labels, {len(task_input.sample_ids)} sets of sample IDs)"
            )

        all_results = []

        species_data = []
        for i, (embeddings, labels, organism, sample_ids) in enumerate(
            zip(
                cell_representation,
                task_input.labels,
                task_input.organisms,
                task_input.sample_ids,
            )
        ):
            embeddings = np.array(embeddings)
            labels = pd.Series(labels)

            logger.info(f"Processing {organism} data: {embeddings.shape} cells")

            # Optionally aggregate cells across donor or sample
            if task_input.aggregation_method != "none":
                if task_input.sample_ids is None:
                    raise ValueError("sample_ids required when aggregation != 'none'")

                embeddings, labels, sample_ids = aggregate_cells_to_samples(
                    embeddings, labels, sample_ids, task_input.aggregation_method
                )
                logger.info(f"Aggregated to {len(embeddings)} samples for {organism}")

            species_data.append((embeddings, labels, str(organism), sample_ids))

        for train_data, test_data in itertools.product(species_data, species_data):
            train_emb, train_labels, train_species, train_sample_ids = train_data
            test_emb, test_labels, test_species, test_sample_ids = test_data

            if train_species == test_species:
                logger.info(
                    f"Running intra-species cross-validation evaluation: {train_species}"
                )
                results = self._run_cross_validation_classification(
                    train_emb,
                    train_labels,
                    train_species,
                    train_sample_ids,
                    n_folds=task_input.n_folds,
                )
                all_results.extend(results)

            else:
                logger.info(
                    f"Running cross-species evaluation: train on {train_species}, test on {test_species}"
                )

                results = self._run_cross_species_classification(
                    train_emb,
                    train_labels,
                    test_emb,
                    test_labels,
                    train_species,
                    test_species,
                )
                all_results.extend(results)

        logger.info(
            f"Completed cross-species evaluation with {len(all_results)} results"
        )

        return CrossSpeciesLabelPredictionOutput(results=all_results)

    def _create_metric_results_for_species_pair(
        self,
        group_df: pd.DataFrame,
        train_species: str,
        test_species: str,
    ) -> List[MetricResult]:
        """Helper to create MetricResult objects for a species pair.

        Args:
            group_df: DataFrame containing results for this species pair
            train_species: Training species name
            test_species: Test species name

        Returns:
            List of MetricResult objects
        """
        logger.debug(
            f"CrossSpeciesLabelPredictionTask._create_metric_results_for_species_pair: group_df shape={group_df.shape}, train_species={train_species}, test_species={test_species}"
        )
        metrics_list = []

        # we have to do some things differently if we average over folds
        is_cross_validation = train_species == test_species

        if is_cross_validation:
            metric_types = {
                "accuracy": MetricType.MEAN_FOLD_ACCURACY,
                "f1": MetricType.MEAN_FOLD_F1_SCORE,
                "precision": MetricType.MEAN_FOLD_PRECISION,
                "recall": MetricType.MEAN_FOLD_RECALL,
                "auroc": MetricType.MEAN_FOLD_AUROC,
            }
        else:
            metric_types = {
                "accuracy": MetricType.ACCURACY,
                "f1": MetricType.F1_SCORE,
                "precision": MetricType.PRECISION,
                "recall": MetricType.RECALL,
                "auroc": MetricType.AUROC,
            }

        # Create aggregated metrics across all classifiers
        base_params = {
            "train_species": train_species,
            "test_species": test_species,
            "classifier": "MEAN(lr,knn,rf)",
        }

        for metric_name, metric_type in metric_types.items():
            metrics_list.append(
                MetricResult(
                    metric_type=metric_type,
                    value=group_df[metric_name].mean(),
                    params=base_params,
                )
            )

        # Create per-classifier metrics
        for clf in group_df["classifier"].unique():
            clf_df = group_df[group_df["classifier"] == clf]
            clf_params = {
                "train_species": train_species,
                "test_species": test_species,
                "classifier": clf,
            }

            for metric_name, metric_type in metric_types.items():
                # For cross-validation, take mean across folds; for cross-species, single value
                value = (
                    clf_df[metric_name].mean()
                    if is_cross_validation
                    else clf_df[metric_name].iloc[0]
                )
                metrics_list.append(
                    MetricResult(
                        metric_type=metric_type,
                        value=value,
                        params=clf_params,
                    )
                )

        return metrics_list

    def _compute_metrics(
        self,
        _: CrossSpeciesLabelPredictionTaskInput,
        task_output: CrossSpeciesLabelPredictionOutput,
    ) -> List[MetricResult]:
        """Compute cross-species label prediction metrics.

        Args:
            _: (unused) Task input
            task_output: Task output containing results

        Returns:
            List of MetricResult objects containing cross-species prediction metrics
        """
        logger.info("Computing cross-species prediction metrics...")
        logger.debug(
            f"CrossSpeciesLabelPredictionTask._compute_metrics: task_output.results shape={len(task_output.results)}"
        )
        results_df = pd.DataFrame(task_output.results)
        metrics_list = []

        # Group by train/test species pairs for aggregation
        for (train_species, test_species), group_df in results_df.groupby(
            ["train_species", "test_species"]
        ):
            species_metrics = self._create_metric_results_for_species_pair(
                group_df,
                train_species,
                test_species,
            )
            metrics_list.extend(species_metrics)

        return metrics_list

    def compute_baseline(
        self,
        expression_data: CellRepresentation,
        baseline_input: NoBaselineInput = None,
    ):
        """Set a baseline for cross-species label prediction.

        Not implemented as standard preprocessing needs to be applied per species.
        """
        raise NotImplementedError(
            "Baseline not implemented for cross-species label prediction"
        )
