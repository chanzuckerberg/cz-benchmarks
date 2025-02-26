from typing import Dict, Set, List
import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    r2_score,
    mean_squared_error,
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from .base import BaseTask
from .utils import cluster_embedding, filter_minimum_class, compute_entropy_per_cell
from ..datasets.single_cell import SingleCellDataset, PerturbationSingleCellDataset
from ..datasets.types import DataType
from scib_metrics import silhouette_batch

logger = logging.getLogger(__name__)


class ClusteringTask(BaseTask):
    def __init__(self, label_key: str):
        self.label_key = label_key

    @property
    def required_inputs(self) -> Set[DataType]:
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}

    def _run_task(self, data: SingleCellDataset):
        adata = data.adata
        adata.obsm["emb"] = data.get_output(DataType.EMBEDDING)
        self.input_labels = data.get_input(DataType.METADATA)[self.label_key]
        self.predicted_labels = cluster_embedding(adata, obsm_key="emb")

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "adjusted_rand_index": adjusted_rand_score(
                self.input_labels, self.predicted_labels
            ),
            "normalized_mutual_info": adjusted_mutual_info_score(
                self.input_labels, self.predicted_labels
            ),
        }


class EmbeddingTask(BaseTask):
    def __init__(self, label_key: str):
        self.label_key = label_key

    @property
    def required_inputs(self) -> Set[DataType]:
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}

    def _run_task(self, data: SingleCellDataset):
        self.embedding = data.get_output(DataType.EMBEDDING)
        self.input_labels = data.get_input(DataType.METADATA)[self.label_key]

    def _compute_metrics(self) -> Dict[str, float]:
        return {"silhouette_score": silhouette_score(self.embedding, self.input_labels)}


class BatchIntegrationTask(BaseTask):
    def __init__(self, label_key: str, batch_key: str):
        self.label_key = label_key
        self.batch_key = batch_key

    @property
    def required_inputs(self) -> Set[DataType]:
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}

    def _run_task(self, data: SingleCellDataset):
        self.embedding = data.get_output(DataType.EMBEDDING)
        self.batch_labels = data.get_input(DataType.METADATA)[self.batch_key]
        self.labels = data.get_input(DataType.METADATA)[self.label_key]

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "entropy_per_cell": compute_entropy_per_cell(
                self.embedding, self.batch_labels
            ),
            "silhouette_score": silhouette_batch(
                self.embedding, self.labels, self.batch_labels
            ),
        }


class MetadataLabelPredictionTask(BaseTask):
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
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}

    def _run_task(self, data: SingleCellDataset):
        logger.info(f"Starting prediction task for label key: {self.label_key}")

        # Get embedding and labels
        embeddings = data.get_output(DataType.EMBEDDING)
        labels = data.get_input(DataType.METADATA)[self.label_key]
        logger.info(
            f"Initial data shape: {embeddings.shape}, labels shape: {labels.shape}"
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
            f"Found {n_classes} classes, using {target_type} averaging for metrics"
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
        logger.info(f"Using {self.n_folds}-fold cross validation with seed {self.seed}")

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

    def _compute_metrics(self) -> Dict[str, float]:
        logger.info("Computing final metrics...")
        results_df = pd.DataFrame(self.results)
        metrics = {}

        # Calculate mean metrics across folds and classifiers
        for metric in ["accuracy", "f1", "precision", "recall"]:
            metrics[f"mean_{metric}"] = results_df[metric].mean()
            logger.info(f"Overall mean {metric}: {metrics[f'mean_{metric}']:.3f}")

            # Add per-classifier means
            for clf in results_df["classifier"].unique():
                clf_results = results_df[results_df["classifier"] == clf]
                metrics[f"{clf}_mean_{metric}"] = clf_results[metric].mean()
                logger.info(
                    f"{clf} mean {metric}: {metrics[f'{clf}_mean_{metric}']:.3f}"
                )

        # Add predictions if generated
        if self.generate_predictions:
            self.predictions_df = pd.DataFrame(self.predictions)
            metrics["predictions"] = self.predictions_df
            logger.info(f"Generated predictions for {len(self.predictions_df)} samples")

        logger.info("Metrics computation completed")
        return metrics


class CrossSpeciesIntegrationTask(BaseTask):
    def __init__(self, label_key: str, species_key: str):
        self.label_key = label_key
        self.species_key = species_key

    @property
    def required_inputs(self) -> Set[DataType]:
        return {DataType.METADATA}

    @property
    def required_outputs(self) -> Set[DataType]:
        return {DataType.EMBEDDING}

    @property
    def requires_multiple_datasets(self) -> bool:
        return True

    def _run_task(self, data: List[SingleCellDataset]):
        self.embedding = np.vstack([d.get_output(DataType.EMBEDDING) for d in data])
        self.labels = np.concatenate(
            [d.get_input(DataType.METADATA)[self.label_key] for d in data]
        )
        self.species = np.concatenate(
            [d.get_input(DataType.METADATA)[self.species_key] for d in data]
        )
        return data

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "entropy_per_cell": compute_entropy_per_cell(self.embedding, self.species),
            "silhouette_score": silhouette_batch(
                self.embedding, self.labels, self.species
            ),
        }

class PerturbationTask(BaseTask):

    @property
    def required_inputs(self) -> Set[DataType]:
        return {DataType.PERTURBATION_TRUTH}

    @property
    def required_outputs(self) -> Set[DataType]:
        return {DataType.PERTURBATION_PRED}

    def _run_task(self, data: PerturbationSingleCellDataset):
        self.perturbation_pred = data.get_output(DataType.PERTURBATION_PRED)
        self.perturbation_truth = data.perturbation_truth
        self.perturbation_ctrl = pd.Series(
            data=data.adata.X.mean(0).A.flatten(),
            index=data.adata.var_names,
            name="ctrl",
        )

    def _compute_metrics(self) -> Dict[str, float]:
        metrics = {}

        avg_perturbation_control = self.perturbation_ctrl

        for key in self.perturbation_pred.keys():
            if key in self.perturbation_truth.keys():
                metrics[key] = {}

                avg_perturbation_pred = self.perturbation_pred[key].mean(axis=0)
                avg_perturbation_truth = self.perturbation_truth[key].mean(axis=0)

                intersecting_genes = list(
                    set(avg_perturbation_pred.index)
                    & set(avg_perturbation_truth.index)
                    & set(avg_perturbation_control.index)
                )

                mse = mean_squared_error(
                    avg_perturbation_pred[intersecting_genes],
                    avg_perturbation_truth[intersecting_genes],
                )
                delta_pearson_corr = r2_score(
                    avg_perturbation_pred[intersecting_genes]
                    - avg_perturbation_control[intersecting_genes],
                    avg_perturbation_truth[intersecting_genes]
                    - avg_perturbation_control[intersecting_genes],
                )
                metrics[key]["mse"] = mse
                metrics[key]["delta_pearson_corr"] = delta_pearson_corr
            else:
                logger.warning(
                    f"Perturbation {key} is not available in the ground truth "
                    "test perturbations. Skipping metrics for this perturbation."
                )

        return metrics
