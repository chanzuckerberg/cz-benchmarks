import logging
from typing import Dict, Set

import pandas as pd
import scipy as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..models.types import ModelType
from ..datasets.base import BaseDataset
from ..datasets.types import DataType
from ..metrics import MetricType, metrics
from .base import BaseTask
from .utils import filter_minimum_class

logger = logging.getLogger(__name__)


class MetadataLabelPredictionTask(BaseTask):
    """Task for predicting labels from embeddings using cross-validation.

    Evaluates multiple classifiers (Logistic Regression, KNN) using k-fold
    cross-validation. Reports standard classification metrics and optionally
    generates per-sample predictions.

    Args:
        label_key: Key to access ground truth labels in metadata
        n_folds: Number of cross-validation folds
        seed: Random seed for reproducibility
        min_class_size: Minimum samples required per class
        generate_predictions: Whether to store per-sample predictions
    """

    def __init__(
        self,
        label_key: str,
        n_folds: int = 5,
        seed: int = 42,
        min_class_size: int = 10,
        generate_predictions: bool = False,
    ):
        self.label_key = label_key
        self.n_folds = n_folds
        self.seed = seed
        self.min_class_size = min_class_size
        self.generate_predictions = generate_predictions
        logger.info(
            "Initialized MetadataLabelPredictionTask with: "
            f"label_key='{label_key}', n_folds={n_folds}, "
            f"min_class_size={min_class_size}, "
            f"generate_predictions={generate_predictions}"
        )

    @property
    def required_inputs(self) -> Set[DataType]:
        """Required input data types.

        Returns:
            Set of required input DataTypes (metadata with labels)
        """
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        """Required output data types.

        Returns:
            required output types from models this task to run  (embedding coordinates)
        """
        return {DataType.EMBEDDING}

    def _run_task(self, data: BaseDataset, model_type: ModelType):
        """Runs cross-validation prediction task.

        Evaluates multiple classifiers using k-fold cross-validation on the
        embedding data. Stores results for metric computation.

        Args:
            data: Dataset containing embedding and ground truth labels
        """
        logger.info(f"Starting prediction task for label key: {self.label_key}")

        # Get embedding and labels
        embeddings = data.get_output(model_type, DataType.EMBEDDING)
        labels = data.get_input(DataType.METADATA)[self.label_key]
        logger.info(
            f"Initial data shape: {embeddings.shape}, " f"labels shape: {labels.shape}"
        )

        # Filter classes with minimum size requirement
        embeddings, labels = filter_minimum_class(
            embeddings, labels, min_class_size=self.min_class_size
        )
        logger.info(f"After filtering: {embeddings.shape} samples remaining")

        # Determine scoring metrics based on number of classes
        n_classes = len(labels.unique())
        target_type = "binary" if n_classes == 2 else "weighted"
        logger.info(
            f"Found {n_classes} classes, using {target_type} " "averaging for metrics"
        )

        scorers = {
            "accuracy": make_scorer(accuracy_score),
            "f1": make_scorer(f1_score, average=target_type),
            "precision": make_scorer(precision_score, average=target_type),
            "recall": make_scorer(recall_score, average=target_type),
        }

        # Setup cross validation
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.seed
        )
        logger.info(
            f"Using {self.n_folds}-fold cross validation " f"with seed {self.seed}"
        )

        # Create classifiers
        classifiers = {
            "lr": Pipeline(
                [("scaler", StandardScaler()), ("lr", LogisticRegression())]
            ),
            "knn": Pipeline(
                [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
            ),
        }
        logger.info(f"Created classifiers: {list(classifiers.keys())}")

        # Store results and predictions
        self.results = []
        self.predictions = []

        # Run cross validation for each classifier
        labels = pd.Categorical(labels.astype(str))
        for name, clf in classifiers.items():
            logger.info(f"Running cross-validation for {name}...")
            cv_results = cross_validate(
                clf,
                embeddings,
                labels.codes,
                cv=skf,
                scoring=scorers,
                return_train_score=False,
            )

            for fold in range(self.n_folds):
                fold_results = {"classifier": name, "split": fold}
                for metric in scorers.keys():
                    fold_results[metric] = cv_results[f"test_{metric}"][fold]
                self.results.append(fold_results)
                logger.debug(f"{name} fold {fold} results: {fold_results}")

            if self.generate_predictions:
                logger.info(f"Generating per-sample predictions for {name}...")
                for fold, (train_idx, test_idx) in enumerate(
                    skf.split(embeddings, labels)
                ):
                    clf.fit(embeddings[train_idx], labels.codes[train_idx])
                    y_pred = clf.predict(embeddings[test_idx])
                    y_true = labels.codes[test_idx]
                    for i, idx in enumerate(test_idx):
                        self.predictions.append(
                            {
                                "classifier": name,
                                "split": fold,
                                "sample_idx": idx,
                                "y_true": labels.categories[y_true[i]],
                                "y_pred": labels.categories[y_pred[i]],
                            }
                        )

        logger.info("Completed cross-validation for all classifiers")

    def _compute_metrics(self) -> Dict[MetricType, float]:
        """Computes classification metrics across all folds.

        Aggregates results from cross-validation and computes mean metrics
        per classifier and overall.

        Returns:
            Dictionary containing mean metrics and optionally predictions
        """
        logger.info("Computing final metrics...")
        results_df = pd.DataFrame(self.results)
        metrics_dict = {}

        # Calculate overall metrics across all classifiers
        metrics_dict["mean_accuracy"] = metrics.compute(
            MetricType.MEAN_FOLD_ACCURACY, results_df=results_df
        )
        metrics_dict["mean_f1"] = metrics.compute(
            MetricType.MEAN_FOLD_F1_SCORE, results_df=results_df
        )
        metrics_dict["mean_precision"] = metrics.compute(
            MetricType.MEAN_FOLD_PRECISION, results_df=results_df
        )
        metrics_dict["mean_recall"] = metrics.compute(
            MetricType.MEAN_FOLD_RECALL, results_df=results_df
        )

        # Calculate per-classifier metrics
        for clf in results_df["classifier"].unique():

            key = f"{clf}_mean_accuracy"
            metrics_dict[key] = metrics.compute(
                MetricType.MEAN_FOLD_ACCURACY, results_df=results_df, classifier=clf
            )

            key = f"{clf}_mean_f1"
            metrics_dict[key] = metrics.compute(
                MetricType.MEAN_FOLD_F1_SCORE, results_df=results_df, classifier=clf
            )

            key = f"{clf}_mean_precision"
            metrics_dict[key] = metrics.compute(
                MetricType.MEAN_FOLD_PRECISION, results_df=results_df, classifier=clf
            )

            key = f"{clf}_mean_recall"
            metrics_dict[key] = metrics.compute(
                MetricType.MEAN_FOLD_RECALL, results_df=results_df, classifier=clf
            )

        # Add predictions if generated
        if self.generate_predictions:
            self.predictions_df = pd.DataFrame(self.predictions)
            metrics_dict["predictions"] = self.predictions_df

        return metrics_dict

    def run_baseline(self, data: BaseDataset):
        """Run a baseline classification using raw gene expression.

        Instead of using embeddings, this method uses the raw gene expression matrix
        as features for classification. This provides a baseline performance to compare
        against embedding-based classification.

        Args:
            data: SingleCellDataset containing AnnData with gene expression and metadata

        Returns:
            Dictionary containing baseline classification metrics
        """

        # Get the AnnData object from the dataset
        adata = data.get_input(DataType.ANNDATA)

        # Extract gene expression matrix
        X = adata.X
        # Convert sparse matrix to dense if needed
        if sp.sparse.issparse(X):
            X = X.toarray()

        # Use raw gene expression as the "embedding" for baseline classification
        data.set_output(ModelType.BASELINE, DataType.EMBEDDING, X)

        # Run the classification task with gene expression features
        baseline_metrics = self.run(data, model_types=[ModelType.BASELINE])[
            ModelType.BASELINE
        ]

        return baseline_metrics
