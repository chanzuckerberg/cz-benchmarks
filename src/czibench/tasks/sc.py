from typing import Dict

from ..datasets.sc import SingleCellDataset
from ..metrics.clustering import adjusted_rand_index, normalized_mutual_info
from ..metrics.embedding import silhouette_score
from .base import BaseTask
from .utils import cluster_embedding


class ClusteringTask(BaseTask):
    def __init__(self, label_key: str):
        self.label_key = label_key

    def validate(self, data: SingleCellDataset):
        return (
            data.output_embedding is not None
            and self.label_key in data.sample_metadata.columns
        )

    def _run_task(self, data: SingleCellDataset) -> SingleCellDataset:
        adata = data.adata
        adata.obsm["emb"] = data.output_embedding
        self.input_labels = data.sample_metadata[self.label_key]
        self.predicted_labels = cluster_embedding(adata, obsm_key="emb")
        return data

    def _compute_metrics(self) -> Dict[str, float]:
        return {
            "adjusted_rand_index": adjusted_rand_index(
                self.input_labels, self.predicted_labels
            ),
            "normalized_mutual_info": normalized_mutual_info(
                self.input_labels, self.predicted_labels
            ),
        }


class EmbeddingTask(BaseTask):
    def __init__(self, label_key: str):
        self.label_key = label_key

    def validate(self, data: SingleCellDataset):
        return (
            data.output_embedding is not None
            and self.label_key in data.sample_metadata.columns
        )

    def _run_task(self, data: SingleCellDataset) -> SingleCellDataset:
        # passthrough, embedding already exists
        self.embedding = data.output_embedding
        self.input_labels = data.sample_metadata[self.label_key]
        return data

    def _compute_metrics(self) -> Dict[str, float]:
        return {"silhouette_score": silhouette_score(self.embedding, self.input_labels)}


class MetadataLabelPredictionTask(BaseTask):
    def __init__(self, label_key: str, n_folds: int = 5, seed: int = 42):
        self.label_key = label_key
        self.n_folds = n_folds
        self.seed = seed

    def validate(self, data: SingleCellDataset):
        return (
            data.output_embedding is not None
            and self.label_key in data.sample_metadata.columns
        )

    def _run_task(self, data: SingleCellDataset) -> SingleCellDataset:
        # Get embedding and labels
        embeddings = data.output_embedding
        labels = data.sample_metadata[self.label_key]

        # Create classifiers
        classifiers = {
            "lr": Pipeline(
                [("scaler", StandardScaler()), ("lr", LogisticRegression())]
            ),
            "knn": Pipeline(
                [("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]
            ),
        }

        # Determine scoring metrics based on number of classes
        target_type = "binary" if len(labels.unique()) == 2 else "weighted"
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

        # Store results
        self.results = []

        # Run cross validation for each classifier
        labels = pd.Categorical(labels.astype(str))
        for name, clf in classifiers.items():
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

        return data

    def _compute_metrics(self) -> Dict[str, float]:
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)

        # Calculate mean metrics across folds and classifiers
        mean_metrics = {}
        for metric in ["accuracy", "f1", "precision", "recall"]:
            mean_metrics[f"mean_{metric}"] = results_df[metric].mean()

            # Add per-classifier means
            for clf in results_df["classifier"].unique():
                clf_results = results_df[results_df["classifier"] == clf]
                mean_metrics[f"{clf}_mean_{metric}"] = clf_results[metric].mean()

        return mean_metrics
